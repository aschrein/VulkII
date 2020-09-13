#include "script.hpp"
#include "utils.hpp"

#include <functional>

#include <SDL.h>
#include <SDL_syswm.h>

#include "rendering.hpp"

#define WIN32_LEAN_AND_MEAN
#include <DirectXMath.h>
#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl.h>
using namespace Microsoft::WRL;

#define DX_ASSERT_OK(x)                                                                            \
  do {                                                                                             \
    HRESULT __res = x;                                                                             \
    if (FAILED(__res)) {                                                                           \
      fprintf(stderr, "__res: %i\n", (i32)__res);                                                  \
      TRAP;                                                                                        \
    }                                                                                              \
  } while (0)

namespace {
struct Slot {
  ID   id;
  i32  frames_referenced;
  ID   get_id() { return id; }
  void set_id(ID _id) { id = _id; }
  void disable() { id._id = 0; }
  bool is_alive() { return id._id != 0; }
  void set_index(u32 index) { id._id = index + 1; }
};

enum class Resource_Type : u32 {
  BUFFER,
  COMMAND_BUFFER,
  IMAGE,
  SHADER,
  SAMPLER,
  PASS,
  BUFFER_VIEW,
  IMAGE_VIEW,
  FENCE,
  SEMAPHORE,
  TIMESTAMP,
  EVENT,
  NONE
};

struct Ref_Cnt : public Slot {
  u32  ref_cnt = 0;
  void rem_reference() {
    ASSERT_DEBUG(ref_cnt > 0);
    ref_cnt--;
  }
  bool is_referenced() { return ref_cnt != 0; }
  void add_reference() { ref_cnt++; }
};

template <typename F> static char const *parse_parentheses(char const *cur, char const *end, F fn) {
  while (cur[0] != '(' && cur != end) cur++;
  if (cur == end) return end;
  cur++;
  return fn(cur, end);
  ;
}

static void execute_preprocessor(List *l, char const *list_end, String_Builder &builder) {
  struct Parameter_Evaluator {
    i32        set;
    i32        binding;
    i32        location;
    i32        array_size;
    string_ref name;
    string_ref type;
    string_ref dim;
    string_ref format;

    void reset() {
      memset(this, 0, sizeof(*this));
      set        = -1;
      binding    = -1;
      location   = -1;
      array_size = -1;
    }
    void exec(List *l) {
      while (l != NULL) {
        if (l->child != NULL) {
          exec(l->child);
        } else if (l->next != NULL) {
          if (l->cmp_symbol("location")) {
            location = l->get(1)->parse_int();
          } else if (l->cmp_symbol("binding")) {
            binding = l->get(1)->parse_int();
          } else if (l->cmp_symbol("array_size")) {
            array_size = l->get(1)->parse_int();
          } else if (l->cmp_symbol("set")) {
            set = l->get(1)->parse_int();
          } else if (l->cmp_symbol("name")) {
            name = l->get(1)->symbol;
          } else if (l->cmp_symbol("type")) {
            type = l->get(1)->symbol;
          } else if (l->cmp_symbol("dim")) {
            dim = l->get(1)->symbol;
          } else if (l->cmp_symbol("format")) {
            format = l->get(1)->symbol;
          }
        }
        l = l->next;
      }
    }
    string_ref format_to_glsl() {
      if (format == stref_s("RGBA32_FLOAT")) {
        return stref_s("rgba32f");
      } else if (format == stref_s("R32_UINT")) {
        return stref_s("r32ui");
      } else if (format == stref_s("RGBA8_SRGBA")) {
        return stref_s("rgba8");
      } else {
        UNIMPLEMENTED;
      }
    }
    string_ref type_to_glsl() {
      // uimage2D
      static char buf[0x100];
      if (format == stref_s("RGBA32_FLOAT")) {
        snprintf(buf, sizeof(buf), "image%.*s", STRF(dim));
        return stref_s(buf);
      } else if (format == stref_s("R32_UINT")) {
        snprintf(buf, sizeof(buf), "uimage%.*s", STRF(dim));
        return stref_s(buf);
      } else if (format == stref_s("RGBA8_SRGBA")) {
        snprintf(buf, sizeof(buf), "image%.*s", STRF(dim));
        return stref_s(buf);
      } else {
        UNIMPLEMENTED;
      }
    }
  };
  Parameter_Evaluator param_eval;
  if (l->child) {
    execute_preprocessor(l->child, list_end, builder);
    return;
  }
  if (l->cmp_symbol("DECLARE_OUTPUT")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(location = %i) out %.*s %.*s;", param_eval.location, STRF(param_eval.type),
                 STRF(param_eval.name));
  } else if (l->cmp_symbol("DECLARE_INPUT")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(location = %i) in %.*s %.*s;", param_eval.location, STRF(param_eval.type),
                 STRF(param_eval.name));
  } else if (l->cmp_symbol("DECLARE_SAMPLER")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(set = %i, binding = %i) uniform sampler %.*s;", param_eval.set,
                 param_eval.binding, STRF(param_eval.name));
  } else if (l->cmp_symbol("DECLARE_IMAGE")) {
    param_eval.reset();
    param_eval.exec(l->next);
    if (param_eval.type == stref_s("SAMPLED")) {
      // uniform layout(binding=2) texture1D g_tTex_unused2;
      // uniform layout(binding=2) sampler g_sSamp3[2];
      // layout(binding = 0, rgba32f) uniform readonly mediump image2D imageM;
      if (param_eval.array_size > 0) {
        builder.putf("layout(set = %i, binding = %i) uniform texture%.*s %.*s[%i];", param_eval.set,
                     param_eval.binding, STRF(param_eval.dim), STRF(param_eval.name),
                     param_eval.array_size);
      } else {
        builder.putf("layout(set = %i, binding = %i) uniform texture%.*s %.*s;", param_eval.set,
                     param_eval.binding, STRF(param_eval.dim), STRF(param_eval.name));
      }
    } else if (param_eval.type == stref_s("WRITE_ONLY")) {
      if (param_eval.array_size > 0) {
        builder.putf("layout(set = %i, binding = %i, %.*s) uniform writeonly "
                     "%.*s %.*s[%i];",
                     param_eval.set, param_eval.binding, STRF(param_eval.format_to_glsl()),
                     STRF(param_eval.type_to_glsl()), STRF(param_eval.name), param_eval.array_size);
      } else {
        builder.putf("layout(set = %i, binding = %i, %.*s) uniform writeonly "
                     "%.*s %.*s;",
                     param_eval.set, param_eval.binding, STRF(param_eval.format_to_glsl()),
                     STRF(param_eval.type_to_glsl()), STRF(param_eval.name));
      }
    } else if (param_eval.type == stref_s("READ_ONLY")) {
      if (param_eval.array_size > 0) {
        builder.putf("layout(set = %i, binding = %i, %.*s) uniform readonly "
                     "%.*s %.*s[%i];",
                     param_eval.set, param_eval.binding, STRF(param_eval.format_to_glsl()),
                     STRF(param_eval.type_to_glsl()), STRF(param_eval.name), param_eval.array_size);
      } else {
        builder.putf("layout(set = %i, binding = %i, %.*s) uniform readonly "
                     "%.*s %.*s;",
                     param_eval.set, param_eval.binding, STRF(param_eval.format_to_glsl()),
                     STRF(param_eval.type_to_glsl()), STRF(param_eval.name));
      }
    } else if (param_eval.type == stref_s("READ_WRITE")) {
      if (param_eval.array_size > 0) {
        builder.putf("layout(set = %i, binding = %i, %.*s) volatile coherent uniform "
                     "%.*s %.*s[%i];",
                     param_eval.set, param_eval.binding, STRF(param_eval.format_to_glsl()),
                     STRF(param_eval.type_to_glsl()), STRF(param_eval.name), param_eval.array_size);
      } else {
        builder.putf("layout(set = %i, binding = %i, %.*s) volatile coherent uniform "
                     "%.*s %.*s;",
                     param_eval.set, param_eval.binding, STRF(param_eval.format_to_glsl()),
                     STRF(param_eval.type_to_glsl()), STRF(param_eval.name));
      }
    } else {
      UNIMPLEMENTED;
    }

  } else if (l->cmp_symbol("GROUP_SIZE")) {
    i32 group_size_x = l->get(1)->parse_int();
    i32 group_size_y = l->get(2)->parse_int();
    i32 group_size_z = l->get(3)->parse_int();
    builder.putf("layout(local_size_x = %i, local_size_y = %i, local_size_z = %i) in;",
                 group_size_x, group_size_y, group_size_z);
  } else if (l->cmp_symbol("ENTRY")) {
    builder.putf("void main() {");
  } else if (l->cmp_symbol("END")) {
    builder.putf("}");
  } else if (l->cmp_symbol("DECLARE_RENDER_TARGET")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(location = %i) out float4 out_rt%i;", param_eval.location,
                 param_eval.location);
  } else if (l->cmp_symbol("EXPORT_POSITION")) {
    string_ref next = l->get(1)->symbol;
    builder.putf("gl_Position = %.*s", (int)(size_t)(list_end - next.ptr - 1), next.ptr);
  } else if (l->cmp_symbol("EXPORT_COLOR")) {
    i32        location;
    string_ref loc_str = l->get(1)->symbol;
    ASSERT_ALWAYS(parse_decimal_int(loc_str.ptr, loc_str.len, &location));
    string_ref next = l->get(2)->symbol;
    builder.putf("out_rt%i = %.*s", location, (int)(size_t)(list_end - next.ptr - 1), next.ptr);
  } else if (l->cmp_symbol("DECLARE_UNIFORM_BUFFER")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(set = %i, binding = %i, std430) uniform UBO_%i_%i {\n", param_eval.set,
                 param_eval.binding, param_eval.set, param_eval.binding);
    List *cur = l->next;
    while (cur != NULL) {
      if (cur->child != NULL && cur->child->cmp_symbol("add_field")) {
        param_eval.reset();
        param_eval.exec(cur->child->next);
        builder.putf("  %.*s %.*s;\n", STRF(param_eval.type), STRF(param_eval.name));
      }
      cur = cur->next;
    }
    builder.putf("};\n");
  } else if (l->cmp_symbol("DECLARE_BUFFER")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(set = %i, binding = %i, scalar) buffer SBO_%i_%i {\n", param_eval.set,
                 param_eval.binding, param_eval.set, param_eval.binding);
    builder.putf("  %.*s %.*s[];\n", STRF(param_eval.type), STRF(param_eval.name));
    builder.putf("};\n");
  } else if (l->cmp_symbol("DECLARE_PUSH_CONSTANTS")) {
    param_eval.reset();
    param_eval.exec(l->next);
    builder.putf("layout(push_constant, std430) uniform PC {\n");
    List *cur = l->next;
    while (cur != NULL) {
      if (cur->child != NULL && cur->child->cmp_symbol("add_field")) {
        param_eval.reset();
        param_eval.exec(cur->child->next);
        builder.putf("  %.*s %.*s;\n", STRF(param_eval.type), STRF(param_eval.name));
      }
      cur = cur->next;
    }
    builder.putf("};\n");
  } else {
    UNIMPLEMENTED;
  }
}

static void preprocess_shader(String_Builder &builder, string_ref body) {
  builder.putf("#version 450\n");
  builder.putf("#extension GL_EXT_nonuniform_qualifier : require\n");
  builder.putf("#extension GL_EXT_scalar_block_layout : require\n");
  builder.putf(R"(
#define float2        vec2
#define float3        vec3
#define float4        vec4
#define float2x2      mat2
#define float3x3      mat3
#define float4x4      mat4
#define int2          ivec2
#define int3          ivec3
#define int4          ivec4
#define uint2         uvec2
#define uint3         uvec3
#define uint4         uvec4
#define float2x2      mat2
#define float3x3      mat3
#define float4x4      mat4
#define u32 uint
#define i32 int
#define f32 float
#define f64 double
#define VERTEX_INDEX  gl_VertexIndex
#define FRAGMENT_COORDINATES  gl_FragCoord
#define INSTANCE_INDEX  gl_InstanceIndex
#define GLOBAL_THREAD_INDEX  gl_GlobalInvocationID
#define GROUPT_INDEX  gl_WorkGroupID
#define LOCAL_THREAD_INDEX  gl_LocalInvocationID
#define lerp          mix
#define float2_splat(x)  vec2(x, x)
#define float3_splat(x)  vec3(x, x, x)
#define float4_splat(x)  vec4(x, x, x, x)
#define bitcast_f32_to_u32(x)  floatBitsToUint(x)
#define bitcast_u32_to_f32(x)  uintBitsToFloat(x)
#define mul4(x, y)  (x * y)

#define image_load(image, coords) imageLoad(image, ivec2(coords))
#define image_store(image, coords, data) imageStore(image, ivec2(coords), data)
#define buffer_load(buffer, index) buffer[index]
#define buffer_store(buffer, index, data) buffer[index] = data
#define buffer_atomic_add(buffer, index, num) atomicAdd(buffer[index], num)
#define buffer_atomic_cas(buffer, index, cmp, replace) atomicCompSwap(buffer[index], cmp, replace)
#define buffer_atomic_exchange(buffer, index, replace) atomicExchange(buffer[index], replace)

)");
  char const *cur       = body.ptr;
  char const *end       = cur + body.len;
  size_t      total_len = 0;
  Pool<List>  list_storage;
  list_storage = Pool<List>::create(1 << 10);
  defer(list_storage.release());

  struct List_Allocator {
    Pool<List> *list_storage;
    List *      alloc() {
      List *out = list_storage->alloc_zero(1);
      return out;
    }
  } list_allocator;
  list_allocator.list_storage = &list_storage;
  auto parse_list             = [&]() {
    list_storage.reset();
    cur = parse_parentheses(cur, end, [&](char const *begin, char const *end) {
      List *root = List::parse(string_ref{begin, (size_t)(end - begin)}, list_allocator, &end);
      if (root == NULL) {
        push_error("Couldn't parse");
        UNIMPLEMENTED;
      }
      execute_preprocessor(root, end, builder);
      return end;
    });
  };
  while (cur != end && total_len < body.len) {
    if (*cur == '@') {
      cur++;
      total_len += 1;
      parse_list();
    } else {
      builder.put_char(*cur);
      cur += 1;
      total_len += 1;
    }
  }
}

template <typename T, typename Parent_t> //
struct Resource_Array {
  Array<T>   items;
  Array<u32> free_items;
  struct Deferred_Release {
    u32 timer;
    u32 item_index;
  };
  Array<Deferred_Release> limbo_items;
  void                    dump() {
    fprintf(stdout, "Resource_Array %s:", Parent_t::NAME);
    fprintf(stdout, "  items: %i", (u32)items.size);
    fprintf(stdout, "  free : %i", (u32)free_items.size);
    fprintf(stdout, "  limbo: %i\n", (u32)limbo_items.size);
  }
  void init() {
    items.init();
    free_items.init();
    limbo_items.init();
  }
  void release() {
    ito(items.size) {
      T &item = items[i];
      if (item.is_alive()) ((Parent_t *)this)->release_item(item);
    }
    items.release();
    free_items.release();
    limbo_items.release();
  }
  void free_slot(ID id) {
    ASSERT_DEBUG(!id.is_null());
    items[id.index()].disable();
    free_items.push(id.index());
  }
  ID push(T t) {
    if (free_items.size) {
      auto id   = free_items.pop();
      items[id] = t;
      items[id].set_index(id);
      return {id + 1};
    }
    items.push(t);
    items.back().set_index(items.size - 1);
    return {(u32)items.size};
  }
  T &operator[](ID id) {
    ASSERT_DEBUG(!id.is_null() && items[id.index()].get_id().index() == id.index());
    return items[id.index()];
  }
  void add_ref(ID id) {
    ASSERT_DEBUG(!id.is_null() && items[id.index()].get_id().index() == id.index());
    items[id.index()].frames_referenced++;
  }
  void remove(ID id, u32 timeout) {
    ASSERT_DEBUG(!id.is_null());
    items[id.index()].disable();
    if (timeout == 0) {
      ((Parent_t *)this)->release_item(items[id.index()]);
      free_items.push(id.index());
    } else {
      limbo_items.push({timeout, id.index()});
    }
  }
  template <typename Ff> void for_each(Ff fn) {
    ito(items.size) {
      T &item = items[i];
      if (item.is_alive()) fn(item);
    }
  }
  void tick() {
    Array<Deferred_Release> new_limbo_items;
    new_limbo_items.init();
    ito(items.size) { items[i].frames_referenced--; }
    ito(limbo_items.size) {
      Deferred_Release &item = limbo_items[i];
      ASSERT_DEBUG(item.timer != 0);
      item.timer -= 1;
      if (item.timer == 0) {
        ((Parent_t *)this)->release_item(items[item.item_index]);
        free_items.push(item.item_index);
      } else {
        new_limbo_items.push(item);
      }
    }
    limbo_items.release();
    limbo_items = new_limbo_items;
  }
};

struct Graphics_Pipeline_State {
  rd::RS_State rs_state;
  rd::DS_State ds_state;
  ID           ps, vs;
  u32          num_rts;
  rd::MS_State ms_state;

  bool operator==(const Graphics_Pipeline_State &that) const {
    return memcmp(this, &that, sizeof(*this)) == 0;
  }
  void reset() {
    memset(this, 0, sizeof(*this)); // Important for memhash
  }
};

u64 hash_of(Graphics_Pipeline_State const &state) {
  return hash_of(string_ref{(char const *)&state, sizeof(state)});
}

struct Window {
  static constexpr u32 MAX_SC_IMAGES = 0x10;
  SDL_Window *         window;

  i32 window_width  = 1280;
  i32 window_height = 720;

  RECT g_WindowRect;

  ComPtr<ID3D12Device2>             device;
  ComPtr<ID3D12CommandQueue>        cmd_queue;
  ComPtr<IDXGISwapChain4>           sc;
  ComPtr<ID3D12Resource>            sc_images[MAX_SC_IMAGES];
  ComPtr<ID3D12GraphicsCommandList> cmd_list;
  ComPtr<ID3D12CommandAllocator>    cmd_allocs[MAX_SC_IMAGES];
  ComPtr<ID3D12DescriptorHeap>      desc_heap;
  UINT                              desc_size;
  UINT                              cur_image;

  ID                                          cur_pass;
  Hash_Table<Graphics_Pipeline_State, ID>     pipeline_cache;
  Hash_Table<ID, ID>                          compute_pipeline_cache;
  Hash_Table<u64, ID>                         shader_cache;
  Hash_Table<rd::Render_Pass_Create_Info, ID> pass_cache;

  void init_ds() {
    shader_cache.init();
    pipeline_cache.init();
    compute_pipeline_cache.init();
  }

  void release() {
    shader_cache.release();
    pipeline_cache.release();
    compute_pipeline_cache.release();

    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  void release_resource(Resource_ID res_id) {
    if (res_id.type == (u32)Resource_Type::PASS) {

    } else if (res_id.type == (u32)Resource_Type::BUFFER) {

    } else if (res_id.type == (u32)Resource_Type::BUFFER_VIEW) {

    } else if (res_id.type == (u32)Resource_Type::IMAGE_VIEW) {

    } else if (res_id.type == (u32)Resource_Type::IMAGE) {

    } else if (res_id.type == (u32)Resource_Type::SHADER) {

    } else if (res_id.type == (u32)Resource_Type::FENCE) {

    } else if (res_id.type == (u32)Resource_Type::EVENT) {

    } else if (res_id.type == (u32)Resource_Type::SEMAPHORE) {

    } else if (res_id.type == (u32)Resource_Type::COMMAND_BUFFER) {

    } else if (res_id.type == (u32)Resource_Type::SAMPLER) {

    } else if (res_id.type == (u32)Resource_Type::TIMESTAMP) {
    } else {
      TRAP;
    }
  }

  void release_swapchain() {}

  void update_swapchain() {
    SDL_SetWindowResizable(window, SDL_FALSE);
    defer(SDL_SetWindowResizable(window, SDL_TRUE));

    release_swapchain();
    u32 format_count = 0;
  }

  void init() {
    init_ds();
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 512, 512,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    TMP_STORAGE_SCOPE;

    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;

#if 1
    ComPtr<ID3D12Debug> debugInterface;
    DX_ASSERT_OK(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
    debugInterface->EnableDebugLayer();
#endif
    update_swapchain();
  }

  void update_surface_size() {}

  void start_frame() {

  restart:
    update_surface_size();
  }
  void end_frame() {}
};

struct Resource_Path {
  u32  set;
  u32  binding;
  u32  element;
  bool operator==(Resource_Path const &that) const {
    return                         //
        set == that.set &&         //
        binding == that.binding && //
        element == that.element;
  }
};

u64 hash_of(Resource_Path const &path) {
  return ::hash_of(path.set) ^ ::hash_of(path.binding) ^ ::hash_of(path.element);
}
} // namespace