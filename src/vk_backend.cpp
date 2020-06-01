#define UTILS_IMPL
#include "utils.hpp"

//#include <imgui.h>
//#include <imgui/examples/imgui_impl_sdl.h>
//#include <imgui/examples/imgui_impl_vulkan.h>

#ifdef __linux__
#define VK_USE_PLATFORM_XCB_KHR
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <shaderc/shaderc.h>
#include <spirv_cross/spirv_cross_c.h>
#include <vulkan/vulkan.h>
#else
#define VK_USE_PLATFORM_WIN32_KHR
#include <SDL.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>
#endif

#include "rendering.hpp"

#define VK_ASSERT_OK(x)                                                                            \
  do {                                                                                             \
    VkResult __res = x;                                                                            \
    if (__res != VK_SUCCESS) {                                                                     \
      fprintf(stderr, "VkResult: %i\n", (i32)__res);                                               \
      TRAP;                                                                                        \
    }                                                                                              \
  } while (0)

Pool<char> string_storage = Pool<char>::create(1 << 20);

VkFormat parse_format(string_ref str) {
  if (str == stref_s("R16_FLOAT")) {
    return VK_FORMAT_R16_SFLOAT;
  } else if (str == stref_s("R32_FLOAT")) {
    return VK_FORMAT_R32_SFLOAT;
  } else if (str == stref_s("RGB32_FLOAT")) {
    return VK_FORMAT_R32G32B32_SFLOAT;
  } else if (str == stref_s("RGBA32_FLOAT")) {
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  } else if (str == stref_s("R32_UINT")) {
    return VK_FORMAT_R32_UINT;
  } else if (str == stref_s("D32_FLOAT")) {
    return VK_FORMAT_D32_SFLOAT;
  } else {
    TRAP;
  }
}

char const *get_glsl_format(VkFormat format) {
  if (format == VK_FORMAT_R16_SFLOAT) {
    return "r16f";
  } else if (format == VK_FORMAT_R32_SFLOAT) {
    return "r32f";
  } else if (format == VK_FORMAT_R32G32_SFLOAT) {
    return "rg32f";
  } else if (format == VK_FORMAT_R32G32B32_SFLOAT) {
    return "rgb32f";
  } else if (format == VK_FORMAT_R32G32B32A32_SFLOAT) {
    return "rgba32f";
  } else {
    TRAP;
  }
}

VkFormat to_vk_format(rd::Format format) {
  switch (format) {
  case rd::Format::R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
  case rd::Format::RG32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
  case rd::Format::RGB32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
  case rd::Format::RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
  case rd::Format::RGB8_SNORM: return VK_FORMAT_R8G8B8_SNORM;
  case rd::Format::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
  case rd::Format::RGB8_SRGBA: return VK_FORMAT_R8G8B8_SRGB;
  case rd::Format::RGBA8_SRGBA: return VK_FORMAT_R8G8B8A8_SRGB;
  case rd::Format::D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
  default: TRAP;
  }
}

struct String_Builder {
  Pool<char> tmp_buf;
  void       init() { tmp_buf = Pool<char>::create(1 << 20); }
  void       release() { tmp_buf.release(); }
  void       reset() { tmp_buf.reset(); }
  string_ref get_str() { return string_ref{(char const *)tmp_buf.at(0), tmp_buf.cursor}; }
  void       putf(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    i32 len = vsprintf(tmp_buf.back(), fmt, args);
    va_end(args);
    ASSERT_ALWAYS(len > 0);
    tmp_buf.advance(len);
  }
  void put_char(char c) { tmp_buf.put(&c, 1); }
};

struct Shader_Builder {
  enum class Freq { PER_DC, PER_FRAME };
  enum class Class_t {
    NONE,
    SRV,
    UAV,
  };
  enum class Global_t {
    SAMPLER,
    BUFFER,
    IMAGE,
  };
  enum class Layout_t {
    NONE,
    ROW_MAJOR,
    COL_MAJOR,
  };
  struct Uniform {
    string_ref name;
    string_ref type;
    u32        count;
    Freq       freq;
    Layout_t   layout;
    bool       operator==(Uniform const &that) {
      return                       //
          name == that.name &&     //
          type == that.type &&     //
          count == that.count &&   //
          layout == that.layout && //
          freq == that.freq;       //
    }
  };
  struct Global {
    string_ref       name;
    Global_t         type;
    Class_t          clazz;
    u32              dim;
    VkFormat         format;
    u32              count;
    Freq             freq;
    VkDescriptorType get_desc_type() {
      if (type == Global_t::BUFFER) {
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      } else if (type == Global_t::IMAGE) {
        if (clazz == Class_t::SRV) {
          return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        } else if (clazz == Class_t::UAV) {
          return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else {
          TRAP;
        }
      } else if (type == Global_t::SAMPLER) {
        return VK_DESCRIPTOR_TYPE_SAMPLER;
      } else {
        TRAP;
      }
    }
    bool operator==(Global const &that) {
      return                       //
          name == that.name &&     //
          clazz == that.clazz &&   //
          type == that.type &&     //
          format == that.format && //
          count == that.count &&   //
          dim == that.dim &&       //
          freq == that.freq;       //
    }
  };
  struct Input {
    string_ref name;
    string_ref type;
    u32        location;
    bool       operator==(Input const &that) {
      return                           //
          name == that.name &&         //
          location == that.location && //
          type == that.type;           //
    }
  };
  struct Output {
    string_ref name;
    string_ref type;
    u32        target;
    bool       operator==(Output const &that) {
      return                   //
          name == that.name && //
          type == that.type;   //
    }
  };
  struct Scratch_State {
    string_ref name;
    string_ref type;
    string_ref layout;
    string_ref clazz;
    VkFormat   format;
    i32        count;
    i32        target;
    Freq       freq;
    i32        dim;
    i32        location;
    void       clear() {
      memset(this, 0, sizeof(*this));
      format = VK_FORMAT_UNDEFINED;
      count  = 1;
      dim    = 1;
      freq   = Freq::PER_DC;
    }
    void eval(List *l) {
      if (l->child != NULL) {
        eval(l->child);
      } else if (l->cmp_symbol("type")) {
        type = l->next->symbol;
      } else if (l->cmp_symbol("class")) {
        clazz = l->next->symbol;
      } else if (l->cmp_symbol("name")) {
        name = l->next->symbol;
      } else if (l->cmp_symbol("layout")) {
        layout = l->next->symbol;
      } else if (l->cmp_symbol("freq")) {
        if (l->next->cmp_symbol("PER_DC")) {
          freq = Freq::PER_DC;
        } else if (l->next->cmp_symbol("PER_FRAME")) {
          freq = Freq::PER_FRAME;
        } else {
          TRAP;
        }
      } else if (l->cmp_symbol("format")) {
        format = parse_format(l->next->symbol);
      } else if (l->cmp_symbol("count")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &count));
      } else if (l->cmp_symbol("location")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &location));
      } else if (l->cmp_symbol("dim")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &dim));
      } else if (l->cmp_symbol("target")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &target));
      }
    }
  } scratch;
  SmallArray<Uniform, 8> uniforms;
  SmallArray<Global, 8>  globals;
  SmallArray<Input, 8>   inputs;
  SmallArray<Output, 8>  outputs;

  u32        dispatch_x, dispatch_y, dispatch_z;
  string_ref body;
  string_ref header;

  shaderc_shader_kind kind;
  void                init() {

    uniforms.init();
    inputs.init();
    outputs.init();
    globals.init();
  }
  void release() {
    uniforms.release();
    inputs.release();
    outputs.release();
    globals.release();
  }

  void eval(List *l) {
    if (l->child != NULL) {
      eval(l->child);
    } else if (l->cmp_symbol("kind")) {
      if (l->next->cmp_symbol("pixel")) {
        kind = shaderc_glsl_fragment_shader;
      } else if (l->next->cmp_symbol("compute")) {
        kind = shaderc_glsl_compute_shader;
      } else if (l->next->cmp_symbol("vertex")) {
        kind = shaderc_glsl_vertex_shader;
      } else {
        TRAP;
      }
    } else if (l->cmp_symbol("create_shader")) {
      List *cur = l->next;
      while (cur != NULL) {
        eval(cur);
        cur = cur->next;
      }
    } else if (l->cmp_symbol("header")) {
      header = l->next->symbol;
    } else if (l->cmp_symbol("body")) {
      body = l->next->symbol;
    } else if (l->cmp_symbol("input")) {
      scratch.clear();
      List *cur = l->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      Input input;
      input.name     = scratch.name;
      input.type     = scratch.type;
      input.location = scratch.location;
      inputs.push(input);
    } else if (l->cmp_symbol("output")) {
      scratch.clear();
      List *cur = l->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      Output output;
      output.name   = scratch.name;
      output.type   = scratch.type;
      output.target = scratch.target;
      outputs.push(output);
    } else if (l->cmp_symbol("uniform")) {
      scratch.clear();
      List *cur = l->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      Uniform uniform;
      uniform.count = scratch.count;
      uniform.freq  = scratch.freq;
      uniform.name  = scratch.name;
      uniform.type  = scratch.type;
      if (scratch.layout == stref_s("row_major")) {
        uniform.layout = Layout_t::ROW_MAJOR;
      } else if (l->next->cmp_symbol("col_major")) {
        uniform.layout = Layout_t::COL_MAJOR;
      } else {
        uniform.layout = Layout_t::NONE;
      }
      uniforms.push(uniform);
    } else if (l->cmp_symbol("global")) {
      scratch.clear();
      List *cur = l->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      Global glob;
      glob.count  = scratch.count;
      glob.format = scratch.format;
      glob.freq   = scratch.freq;
      glob.name   = scratch.name;
      glob.dim    = scratch.dim;
      if (scratch.clazz == stref_s("SRV")) {
        glob.clazz = Class_t::SRV;
      } else if (l->next->cmp_symbol("UAV")) {
        glob.clazz = Class_t::UAV;
      } else {
        TRAP;
      }
      if (scratch.type == stref_s("image")) {
        glob.type = Global_t::IMAGE;
      } else if (l->next->cmp_symbol("buffer")) {
        glob.type = Global_t::BUFFER;
      } else if (l->next->cmp_symbol("sampler")) {
        glob.type = Global_t::SAMPLER;
      } else {
        TRAP;
      }
      globals.push(glob);
    }
  }
};

string_ref relocate_cstr(string_ref old) {
  char *     new_ptr = string_storage.put(old.ptr, old.len + 1);
  string_ref new_ref = string_ref{new_ptr, old.len};
  new_ptr[old.len]   = '\0';
  return new_ref;
}

struct Slot {
  ID   id;
  ID   get_id() { return id; }
  void set_id(ID _id) { id = _id; }
  void disable() { id._id = 0; }
  bool is_alive() { return id._id != 0; }
  void set_index(u32 index) { id._id = index + 1; }
};

enum class Resource_Type { BUFFER, TEXTURE, RT, NONE };

struct Resource_Desc : public Slot {
  Resource_Type type;
  u32           ref;
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

// 1 <<  8 = alignment
// 1 << 17 = num of blocks
struct Mem_Chunk : public Ref_Cnt {
  VkDeviceMemory        mem              = VK_NULL_HANDLE;
  VkMemoryPropertyFlags prop_flags       = 0;
  static constexpr u32  PAGE_SIZE        = 0x1000;
  u32                   size             = 0;
  u32                   cursor           = 0; // points to the next free 4kb byte block
  u32                   memory_type_bits = 0;
  void                  dump() {
    fprintf(stdout, "Mem_Chunk {\n");
    fprintf(stdout, "  ref_cnt: %i\n", ref_cnt);
    fprintf(stdout, "  size   : %i\n", size);
    fprintf(stdout, "  cursor : %i\n", cursor);
    fprintf(stdout, "}\n");
  }
  void init(VkDevice device, u32 num_pages, u32 heap_index, VkMemoryPropertyFlags prop_flags,
            u32 type_bits) {
    VkMemoryAllocateInfo info;
    MEMZERO(info);
    info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize  = num_pages * PAGE_SIZE;
    info.memoryTypeIndex = heap_index;
    VK_ASSERT_OK(vkAllocateMemory(device, &info, nullptr, &mem));
    this->size             = num_pages;
    this->prop_flags       = prop_flags;
    this->memory_type_bits = type_bits;
    this->cursor           = 0;
  }

  void release(VkDevice device) { vkFreeMemory(device, mem, NULL); }
  bool has_space(u32 req_size) {
    if (ref_cnt == 0) cursor = 0;
    return cursor + ((req_size + PAGE_SIZE - 1) / PAGE_SIZE) < size;
  }
  u32 alloc(u32 alignment, u32 req_size) {
    ASSERT_DEBUG((alignment & (alignment - 1)) == 0); // PoT
    ASSERT_DEBUG(((alignment - 1) & PAGE_SIZE) == 0); // 4kb bytes is enough to align
    u32 offset = cursor;
    cursor += ((req_size + PAGE_SIZE - 1) / PAGE_SIZE);
    ASSERT_DEBUG(cursor < size);
    ref_cnt++;
    return offset * PAGE_SIZE;
  }
};

struct Buffer : public Ref_Cnt {
  ID                 mem_chunk_id;
  u32                mem_offset;
  VkBuffer           buffer;
  VkBufferCreateInfo create_info;
  VkAccessFlags      access_flags;
};

struct Image : public Ref_Cnt {
  ID                 mem_chunk_id;
  u32                mem_offset;
  VkImageLayout      layout;
  VkAccessFlags      access_flags;
  VkImageAspectFlags aspect;
  VkImage            image;
  VkImageCreateInfo  create_info;
};

struct ImageView : public Slot {
  ID                    img_id;
  VkImageView           view;
  VkImageViewCreateInfo create_info;
};

struct BufferView : public Slot {
  ID                     buf_id;
  VkBufferView           view;
  VkBufferViewCreateInfo create_info;
};

struct RT : public Slot {
  ID img_id;
  ID view_id;
};

// struct Shader_Descriptor {
//  u32                          set;
//  VkDescriptorSetLayoutBinding layout;
//};

// struct Attribute_Source {
//  uint32_t binding;
//  uint32_t offset;
//  VkFormat format;
//};

struct Pass_Input {
  string_ref name;
  bool       history;
  void       relocate() { name = relocate_cstr(name); }
};

struct RT_Info {
  VkFormat format;
  rd::RT_t type;
};

struct Render_Pass : public Slot {
  string_ref                name;
  SmallArray<Pass_Input, 4> deps;
  SmallArray<string_ref, 4> rts;
  u32                       width;
  u32                       height;
  Resource_ID               depth_target;
  VkRenderPass              pass;
  VkFramebuffer             fb;
  List *                    src;

  void init() {
    memset(this, 0, sizeof(*this));
    deps.init();
    rts.init();
  }

  void release(VkDevice device) {
    vkDestroyRenderPass(device, pass, NULL);
    vkDestroyFramebuffer(device, fb, NULL);
    deps.release();
    rts.release();
    memset(this, 0, sizeof(*this));
  }

  void relocate() {
    name = relocate_cstr(name);
    ito(deps.size) deps[i].relocate();
    ito(rts.size) rts[i] = relocate_cstr(rts[i]);
  }
};

struct Graphics_Pipeline_State {
  VkVertexInputBindingDescription   bindings[0x10];
  u32                               num_bindings;
  VkVertexInputAttributeDescription attributes[0x10];
  u32                               num_attributes;
  VkCullModeFlags                   cull_mode;
  VkFrontFace                       front_face;
  VkPolygonMode                     polygon_mode;
  float                             line_width;
  bool                              enable_depth_test;
  VkCompareOp                       cmp_op;
  bool                              enable_depth_write;
  float                             max_depth;
  VkPrimitiveTopology               topology;
  float                             depth_bias_const;
  ID                                ps, vs;
  u64                               ps_hash, vs_hash;
  ID                                pass;
  u64                               dummy; // used for hashing to emulate C string
  bool                              operator==(const Graphics_Pipeline_State &that) const {
    return memcmp(this, &that, sizeof(*this)) == 0;
  }
  void reset() {
    memset(this, 0, sizeof(*this)); // Important for memhash
  }
};

u64 hash_of(Graphics_Pipeline_State const &state) { return hash_of((char const *)&state); }

struct Shader : public Ref_Cnt {
  List *root;
  //  VkShaderModule module;
  u64 hash;
  //  struct Vertex_Attribute {
  //    string_ref name;
  //    VkFormat   format;
  //    u32        location;
  //  };
  //  struct Shader_Input {
  //    string_ref name;
  //    VkFormat   format;
  //    u32        location;
  //  };
  //  struct Resource_Slot {
  //    string_ref name;
  //    u32        size;
  //    u32        set;
  //    u32        binding;
  //  };
  //  SmallArray<Vertex_Attribute, 4> attributes;
  //  SmallArray<Shader_Input, 4>     inputs;
  //  SmallArray<Shader_Input, 4>     outputs;
  //  SmallArray<Resource_Slot, 4>    resources;
};

struct Graphics_Pipeline_Wrapper : public Slot {
  VkDescriptorSetLayout                                           set_layouts[4];
  u32                                                             num_set_layouts;
  VkPipelineLayout                                                pipeline_layout;
  VkPipeline                                                      pipeline;
  Hash_Table<string_ref, Pair<u32, u32>, Default_Allocator, 0x10> global_slots;
  Hash_Table<string_ref, u32, Default_Allocator, 0x10>            uniform_slots;
  VkShaderModule                                                  vs_module;
  VkShaderModule                                                  ps_module;
  u32                                                             uniform_size;
  u32                                                             push_constants_size;

  void release(VkDevice device) {
    ito(num_set_layouts) vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, vs_module, NULL);
    vkDestroyShaderModule(device, ps_module, NULL);
    global_slots.release();
    uniform_slots.release();
  }

  static VkShaderModule compile_glsl(VkDevice device, string_ref text, shaderc_shader_kind kind) {
    shaderc_compiler_t        compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t options  = shaderc_compile_options_initialize();
    shaderc_compile_options_set_source_language(options, shaderc_source_language_glsl);
    shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_3);
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan,
                                           shaderc_env_version_vulkan_1_2);
    shaderc_compilation_result_t result =
        shaderc_compile_into_spv(compiler, text.ptr, text.len, kind, "tmp.lsp", "main", options);
    defer({
      shaderc_result_release(result);
      shaderc_compiler_release(compiler);
      shaderc_compile_options_release(options);
    });
    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
      push_error("%.*s\n", STRF(text));
      push_error(shaderc_result_get_error_message(result));
      TRAP;
    }
    size_t                   len      = shaderc_result_get_length(result);
    u32 *                    bytecode = (u32 *)shaderc_result_get_bytes(result);
    VkShaderModuleCreateInfo info;
    MEMZERO(info);
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = len;
    info.flags    = 0;
    info.pCode    = bytecode;
    VkShaderModule module;
    VK_ASSERT_OK(vkCreateShaderModule(device, &info, NULL, &module));
    return module;
  }

  static void print_global(String_Builder &builder, Shader_Builder::Global &glob, u32 &set_counter,
                           u32 &binding_counter) {
    if (glob.type == Shader_Builder::Global_t::IMAGE) {
      if (glob.clazz == Shader_Builder::Class_t::SRV) {
        if (glob.dim == 2) {
          if (glob.count == 1) {
            builder.putf("layout(set = %i, binding = %i) texture2D %.*s;\n", set_counter,
                         binding_counter, STRF(glob.name));
            binding_counter += 1;
          } else {
            ASSERT_ALWAYS(glob.count > 1);
            builder.putf("layout(set = %i, binding = %i) texture2D %.*s[%i];\n", set_counter,
                         binding_counter, STRF(glob.name), glob.count);
            binding_counter += glob.count;
          }
        } else {
          TRAP;
        }
      } else if (glob.clazz == Shader_Builder::Class_t::UAV) {
        if (glob.dim == 2) {
          if (glob.count == 1) {
            builder.putf("layout(set = %i, binding = %i, %s) image2D %.*s;\n", set_counter,
                         binding_counter, get_glsl_format(glob.format), STRF(glob.name));
            binding_counter += 1;
          } else {
            ASSERT_ALWAYS(glob.count > 1);
            builder.putf("layout(set = %i, binding = %i, %s) image2D %.*s[%i];\n", set_counter,
                         binding_counter, get_glsl_format(glob.format), STRF(glob.name),
                         glob.count);
            binding_counter += glob.count;
          }
        } else {
          TRAP;
        }
      } else {
        TRAP;
      }
    } else {
      TRAP;
    }
  }

  static void preprocess_shader(String_Builder &builder, Shader_Builder &shb) {
    char const *cur       = shb.body.ptr;
    char const *end       = cur + shb.body.len;
    size_t      total_len = 0;
    while (cur != end && total_len < shb.body.len) {
      if (*cur == '@') {
        cur++;
        total_len += 1;
        i32 match_len = -1;
        if ((match_len = str_match(cur, "EXPORT_COLOR0<")) > 0) {
          cur += match_len;
          total_len += match_len;
          i32 symbol_len = str_find(cur, 0x100, '>');
          ASSERT_ALWAYS(symbol_len > 0);
          builder.putf("_rt0 = %.*s;", symbol_len, cur);
          cur += symbol_len + 1;
          total_len += symbol_len + 1;
        } else if ((match_len = str_match(cur, "EXPORT_POSITION<")) > 0) {
          cur += match_len;
          total_len += match_len;
          i32 symbol_len = str_find(cur, 0x100, '>');
          ASSERT_ALWAYS(symbol_len > 0);
          builder.putf("gl_Position = %.*s;", symbol_len, cur);
          cur += symbol_len + 1;
          total_len += symbol_len + 1;
        } else if ((match_len = str_match(cur, "ENTRY")) > 0) {
          cur += match_len;
          total_len += match_len;
          builder.putf("void main()");
        } else if ((match_len = str_match(cur, "GLOBAL<")) > 0) {
          cur += match_len;
          total_len += match_len;
          i32 symbol_len = str_find(cur, 0x100, '>');
          ASSERT_ALWAYS(symbol_len > 0);
          builder.putf("%.*s", symbol_len, cur);
          cur += symbol_len + 1;
          total_len += symbol_len + 1;
        } else if ((match_len = str_match(cur, "UNIFORM<")) > 0) {
          cur += match_len;
          total_len += match_len;
          i32 symbol_len = str_find(cur, 0x100, '>');
          ASSERT_ALWAYS(symbol_len > 0);
          builder.putf("ubo.%.*s", symbol_len, cur);
          cur += symbol_len + 1;
          total_len += symbol_len + 1;
        } else {
          TRAP;
        }
      } else {
        builder.put_char(*cur);
        cur += 1;
        total_len += 1;
      }
    }
  }

  static u32 parse_size(string_ref type) {
    if (                             //
        type == stref_s("float") ||  //
        type == stref_s("float2") || //
        type == stref_s("float3") || //
        type == stref_s("float4") || //
        type == stref_s("int") ||    //
        type == stref_s("int2") ||   //
        type == stref_s("int3") ||   //
        type == stref_s("int4") ||   //
        type == stref_s("uint") ||   //
        type == stref_s("uint2") ||  //
        type == stref_s("uint3") ||  //
        type == stref_s("uint4")     //
    ) {
      return 16;
    } else if (type == stref_s("float2x2")) {
      return 16 * 2;
    } else if (type == stref_s("float3x3")) {
      return 16 * 3;
    } else if (type == stref_s("float4x4")) {
      return 16 * 4;
    } else {
      TRAP;
    }
  }

  void init(VkDevice                 device,    //
            Render_Pass &            pass,      //
            Shader &                 vs_shader, //
            Shader &                 ps_shader, //
            Graphics_Pipeline_State &pipeline_info) {
    uniform_slots.init();
    global_slots.init();
    (void)pipeline_info;
    SmallArray<VkDescriptorSetLayoutBinding, 8> set_bindings;
    set_bindings.init();
    defer(set_bindings.release());
    {
      Shader_Builder vsbuilder;
      vsbuilder.init();
      defer(vsbuilder.release());
      Shader_Builder psbuilder;
      psbuilder.init();
      defer(psbuilder.release());

      vsbuilder.eval(vs_shader.root);
      psbuilder.eval(ps_shader.root);
      ASSERT_ALWAYS(vsbuilder.outputs.size == psbuilder.inputs.size);
      Hash_Table<string_ref, Shader_Builder::Uniform> uniform_table;
      Hash_Table<string_ref, Shader_Builder::Global>  global_table;
      uniform_table.init();
      global_table.init();
      defer({
        global_table.release();
        uniform_table.release();
      });
      ito(vsbuilder.uniforms.size) {
        auto &uniform = vsbuilder.uniforms[i];
        uniform_table.insert(uniform.name, uniform);
      }
      ito(psbuilder.uniforms.size) {
        Shader_Builder::Uniform &uniform = psbuilder.uniforms[i];
        if (uniform_table.contains(uniform.name)) {
          ASSERT_ALWAYS(uniform == uniform_table.get(uniform.name));
        } else {
          uniform_table.insert(uniform.name, uniform);
        }
      }
      ito(vsbuilder.globals.size) {
        Shader_Builder::Global &glob = vsbuilder.globals[i];
        global_table.insert(glob.name, glob);
      }
      ito(psbuilder.globals.size) {
        auto &glob = psbuilder.globals[i];
        if (global_table.contains(glob.name)) {
          ASSERT_ALWAYS(glob == global_table.get(glob.name));
        } else {
          global_table.insert(glob.name, glob);
        }
      }

      String_Builder common_header;
      String_Builder ps_text;
      String_Builder vs_text;
      ps_text.init();
      vs_text.init();
      common_header.init();
      defer({
        common_header.release();
        ps_text.release();
        vs_text.release();
      });
      u32 set_counter     = 0;
      u32 binding_counter = 0;
      common_header.putf("#version 450\n");
      common_header.putf("#extension GL_EXT_nonuniform_qualifier : require\n");
      common_header.putf(R"(
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define uint2 uvec2
#define uint3 uvec3
#define uint4 uvec4
#define float2x2 mat2
#define float3x3 mat3
#define float4x4 mat4

)");
      { // One uniform buffer
        VkDescriptorSetLayoutBinding set_binding;
        set_binding.binding            = 0;
        set_binding.descriptorCount    = 1;
        set_binding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        set_binding.pImmutableSamplers = NULL;
        set_binding.stageFlags         = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
        set_bindings.push(set_binding);
      }
      u32 uniform_offset = 0;
      common_header.putf("layout(set = %i, binding = %i, std140) uniform UBO_t {\n", set_counter,
                         binding_counter);
      uniform_table.iter_values([&](Shader_Builder::Uniform &uniform) {
        ASSERT_ALWAYS(uniform.count == 1);
        ASSERT_ALWAYS(uniform.layout == Shader_Builder::Layout_t::NONE);
        uniform_slots.insert(uniform.name, uniform_offset);
        common_header.putf("  %.*s %.*s;\n", STRF(uniform.type), STRF(uniform.name));
        uniform_offset += parse_size(uniform.type);
      });
      common_header.putf("} ubo;\n");
      binding_counter += 1;
      global_table.iter_values([&](Shader_Builder::Global &glob) {
        {
          ASSERT_ALWAYS(set_counter == 0 && "One set for now");
          VkDescriptorSetLayoutBinding set_binding;
          set_binding.binding            = binding_counter;
          set_binding.descriptorCount    = glob.count;
          set_binding.descriptorType     = glob.get_desc_type();
          set_binding.pImmutableSamplers = NULL;
          set_binding.stageFlags         = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
          set_bindings.push(set_binding);
        }
        global_slots.insert(relocate_cstr(glob.name), Pair<u32, u32>{set_counter, binding_counter});
        print_global(common_header, glob, set_counter, binding_counter);
      });
      string_ref header = common_header.get_str();
      ps_text.putf("%.*s", STRF(header));
      vs_text.putf("%.*s", STRF(header));
      ito(psbuilder.inputs.size) {
        auto &input = psbuilder.inputs[i];
        ps_text.putf("layout(location = %i) in %.*s %.*s;\n", i, STRF(input.type),
                     STRF(input.name));
      }
      ito(psbuilder.outputs.size) {
        auto &output = psbuilder.outputs[i];
        ps_text.putf("layout(location = %i) out %.*s _rt%i;\n", i, STRF(output.type),
                     output.target);
      }
      ito(vsbuilder.inputs.size) {
        auto &input = vsbuilder.inputs[i];
        vs_text.putf("layout(location = %i) in %.*s %.*s;\n", input.location, STRF(input.type),
                     STRF(input.name));
      }
      ito(vsbuilder.outputs.size) {
        auto &output = vsbuilder.outputs[i];
        vs_text.putf("layout(location = %i) out %.*s %.*s;\n", i, STRF(output.type),
                     STRF(output.name));
      }
      preprocess_shader(ps_text, psbuilder);
      preprocess_shader(vs_text, vsbuilder);
      vs_module = compile_glsl(device, vs_text.get_str(), shaderc_glsl_vertex_shader);
      ps_module = compile_glsl(device, ps_text.get_str(), shaderc_glsl_fragment_shader);
    }
    VkPipelineShaderStageCreateInfo stages[2];
    {
      VkPipelineShaderStageCreateInfo stage;
      stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module = vs_module;
      stage.pName  = "main";
      stages[0]    = stage;
    }
    {
      VkPipelineShaderStageCreateInfo stage;
      stage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
      stage.module = ps_module;
      stage.pName  = "main";
      stages[1]    = stage;
    }
    {
      VkDescriptorSetLayoutBindingFlagsCreateInfo binding_infos;
      SmallArray<VkDescriptorBindingFlags, 8>     binding_flags;
      binding_flags.init();
      defer(binding_flags.release());
      ito(set_bindings.size) {
        if (set_bindings[i].descriptorCount > 1) {
          binding_flags.push(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        } else {
          binding_flags.push(0);
        }
      }
      binding_infos.bindingCount  = binding_flags.size;
      binding_infos.pBindingFlags = &binding_flags[0];
      binding_infos.pNext         = NULL;
      binding_infos.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

      VkDescriptorSetLayoutCreateInfo set_layout_create_info;
      MEMZERO(set_layout_create_info);
      set_layout_create_info.bindingCount = set_bindings.size;
      set_layout_create_info.pBindings    = &set_bindings[0];
      set_layout_create_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
      set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      set_layout_create_info.pNext = (void *)&binding_infos;
      VkDescriptorSetLayout set_layout;
      VK_ASSERT_OK(vkCreateDescriptorSetLayout(device, &set_layout_create_info, NULL, &set_layout));
      set_layouts[num_set_layouts++] = set_layout;
    }
    // VkPushConstantRange push_range;
    // MEMZERO(push_range);
    // push_range.offset = 0;
    // push_range.size = ???
    {
      VkPipelineLayoutCreateInfo pipe_layout_info;
      MEMZERO(pipe_layout_info);
      pipe_layout_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipe_layout_info.pSetLayouts    = &set_layouts[0];
      pipe_layout_info.setLayoutCount = num_set_layouts;
      VK_ASSERT_OK(vkCreatePipelineLayout(device, &pipe_layout_info, NULL, &pipeline_layout));
    }
    {
      VkGraphicsPipelineCreateInfo info;
      MEMZERO(info);
      info.layout = pipeline_layout;
      SmallArray<VkPipelineColorBlendAttachmentState, 4> blend_states;
      blend_states.init();
      defer({ blend_states.release(); });
      // @TODO Add blend states
      ito(pass.rts.size) {
        VkPipelineColorBlendAttachmentState blend_state;
        MEMZERO(blend_state);
        blend_state.colorWriteMask =   //
            VK_COLOR_COMPONENT_R_BIT | //
            VK_COLOR_COMPONENT_G_BIT | //
            VK_COLOR_COMPONENT_B_BIT | //
            VK_COLOR_COMPONENT_A_BIT;  //
        blend_state.blendEnable = VK_FALSE;
        blend_states.push(blend_state);
      }
      VkPipelineColorBlendStateCreateInfo blend_create_info;
      MEMZERO(blend_create_info);
      blend_create_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      blend_create_info.attachmentCount = pass.rts.size;
      blend_create_info.logicOpEnable   = VK_FALSE;
      blend_create_info.pAttachments    = &blend_states[0];
      info.pColorBlendState             = &blend_create_info;

      VkPipelineDepthStencilStateCreateInfo ds_create_info;
      MEMZERO(ds_create_info);
      ds_create_info.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
      ds_create_info.depthTestEnable  = pipeline_info.enable_depth_test ? VK_TRUE : VK_FALSE;
      ds_create_info.depthCompareOp   = pipeline_info.cmp_op;
      ds_create_info.depthWriteEnable = pipeline_info.enable_depth_write;
      ds_create_info.maxDepthBounds   = pipeline_info.max_depth;
      info.pDepthStencilState         = &ds_create_info;

      VkPipelineRasterizationStateCreateInfo rs_create_info;
      MEMZERO(rs_create_info);
      rs_create_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
      rs_create_info.cullMode        = pipeline_info.cull_mode;
      rs_create_info.frontFace       = pipeline_info.front_face;
      rs_create_info.lineWidth       = pipeline_info.line_width;
      rs_create_info.polygonMode     = pipeline_info.polygon_mode;
      rs_create_info.depthBiasEnable = pipeline_info.depth_bias_const != 0.0f;
      rs_create_info.depthBiasConstantFactor = pipeline_info.depth_bias_const;
      info.pRasterizationState               = &rs_create_info;

      VkDynamicState dynamic_states[] = {
          VK_DYNAMIC_STATE_VIEWPORT,
          VK_DYNAMIC_STATE_SCISSOR,
      };
      VkPipelineDynamicStateCreateInfo dy_create_info;
      MEMZERO(dy_create_info);
      dy_create_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dy_create_info.dynamicStateCount = ARRAY_SIZE(dynamic_states);
      dy_create_info.pDynamicStates    = dynamic_states;
      info.pDynamicState               = &dy_create_info;

      VkPipelineMultisampleStateCreateInfo ms_state;
      MEMZERO(ms_state);
      ms_state.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
      ms_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      info.pMultisampleState        = &ms_state;

      VkPipelineInputAssemblyStateCreateInfo ia_create_info;
      MEMZERO(ia_create_info);
      ia_create_info.sType     = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      ia_create_info.topology  = pipeline_info.topology;
      info.pInputAssemblyState = &ia_create_info;

      VkPipelineVertexInputStateCreateInfo vs_create_info;
      MEMZERO(vs_create_info);
      vs_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vs_create_info.pVertexAttributeDescriptions    = pipeline_info.attributes;
      vs_create_info.vertexAttributeDescriptionCount = pipeline_info.num_attributes;
      vs_create_info.pVertexBindingDescriptions      = pipeline_info.bindings;
      vs_create_info.vertexAttributeDescriptionCount = pipeline_info.num_bindings;
      info.pVertexInputState                         = &vs_create_info;

      info.renderPass = pass.pass;

      VK_ASSERT_OK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
    }
  }
};

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

struct Window {
  static constexpr u32 MAX_SC_IMAGES = 0x10;
  SDL_Window *         window        = 0;

  VkSurfaceKHR surface       = VK_NULL_HANDLE;
  i32          window_width  = 1280;
  i32          window_height = 720;

  VkInstance       instance                   = VK_NULL_HANDLE;
  VkPhysicalDevice physdevice                 = VK_NULL_HANDLE;
  VkQueue          queue                      = VK_NULL_HANDLE;
  VkDevice         device                     = VK_NULL_HANDLE;
  VkCommandPool    cmd_pool                   = VK_NULL_HANDLE;
  VkCommandBuffer  cmd_buffers[MAX_SC_IMAGES] = {};

  VkSwapchainKHR     swapchain                      = VK_NULL_HANDLE;
  VkRenderPass       sc_render_pass                 = VK_NULL_HANDLE;
  uint32_t           sc_image_count                 = 0;
  VkImageLayout      sc_image_layout[MAX_SC_IMAGES] = {};
  VkImage            sc_images[MAX_SC_IMAGES]       = {};
  VkImageView        sc_image_views[MAX_SC_IMAGES]  = {};
  VkFramebuffer      sc_framebuffers[MAX_SC_IMAGES] = {};
  VkExtent2D         sc_extent                      = {};
  VkSurfaceFormatKHR sc_format                      = {};

  u32         frame_id                         = 0;
  u32         cmd_index                        = 0;
  u32         image_index                      = 0;
  VkFence     frame_fences[MAX_SC_IMAGES]      = {};
  VkSemaphore sc_free_sem[MAX_SC_IMAGES]       = {};
  VkSemaphore render_finish_sem[MAX_SC_IMAGES] = {};

  u32 graphics_queue_id = 0;
  u32 compute_queue_id  = 0;
  u32 transfer_queue_id = 0;

  Array<Mem_Chunk> mem_chunks;
  struct Buffer_Array : Resource_Array<Buffer, Buffer_Array> {
    static constexpr char const NAME[] = "Buffer_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Buffer &buf) {
      vkDestroyBuffer(wnd->device, buf.buffer, NULL);
      wnd->mem_chunks[buf.mem_chunk_id.index()].rem_reference();
      MEMZERO(buf);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } buffers;
  struct Image_Array : Resource_Array<Image, Image_Array> {
    static constexpr char const NAME[] = "Image_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Image &img) {
      vkDestroyImage(wnd->device, img.image, NULL);
      wnd->mem_chunks[img.mem_chunk_id.index()].rem_reference();
      MEMZERO(img);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } images;
  struct BufferView_Array : Resource_Array<BufferView, BufferView_Array> {
    static constexpr char const NAME[] = "BufferView_Array";
    Window *                    wnd    = NULL;
    void                        release_item(BufferView &buf) {
      vkDestroyBufferView(wnd->device, buf.view, NULL);
      wnd->buffers[buf.buf_id].rem_reference();
      MEMZERO(buf);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } buffer_views;
  struct ImageView_Array : Resource_Array<ImageView, ImageView_Array> {
    static constexpr char const NAME[] = "ImageView_Array";
    Window *                    wnd    = NULL;
    void                        release_item(ImageView &img) {
      vkDestroyImageView(wnd->device, img.view, NULL);
      wnd->images[img.img_id].rem_reference();
      MEMZERO(img);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } image_views;
  struct Shader_Array : Resource_Array<Shader, Shader_Array> {
    static constexpr char const NAME[] = "Shader_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Shader &shader) { (void)shader; }
    void                        init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } shaders;
  struct Render_Pass_Array : Resource_Array<Render_Pass, Render_Pass_Array> {
    static constexpr char const NAME[] = "Render_Pass_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Render_Pass &item) { item.release(wnd->device); }
    void                        init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } render_passes;
  struct RT_Array : Resource_Array<RT, RT_Array> {
    static constexpr char const NAME[] = "RT_Array";
    Window *                    wnd    = NULL;
    void                        release_item(RT &item) {
      wnd->images[item.img_id].rem_reference();
      wnd->image_views.remove(item.view_id, 3);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } rts;
  struct Pipe_Array : Resource_Array<Graphics_Pipeline_Wrapper, Pipe_Array> {
    static constexpr char const NAME[] = "Pipe_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Graphics_Pipeline_Wrapper &item) {
      item.release(wnd->device);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } pipelines;
  ID                                                             cur_pass;
  Hash_Table<string_ref, ID>                                     named_render_passes;
  Graphics_Pipeline_State                                        graphics_state;
  Hash_Table<string_ref, Resource_ID>                            named_resources;
  Hash_Table<Graphics_Pipeline_State, Graphics_Pipeline_Wrapper> pipeline_cache;

  void init_ds() {
    named_resources.init();
    pipeline_cache.init();
    mem_chunks.init();
    named_render_passes.init();
    buffers.init(this);
    images.init(this);
    shaders.init(this);
    buffer_views.init(this);
    image_views.init(this);
    render_passes.init(this);
    rts.init(this);
    pipelines.init(this);
  }

  void release() {
    named_resources.release();
    pipeline_cache.release();
    buffers.release();
    images.release();
    named_render_passes.release();
    shaders.release();
    buffer_views.release();
    image_views.release();
    render_passes.release();
    rts.release();
    pipelines.release();
    ito(mem_chunks.size) mem_chunks[i].release(device);
    mem_chunks.release();
    vkDeviceWaitIdle(device);
    ito(sc_image_count) vkDestroySemaphore(device, sc_free_sem[i], NULL);
    ito(sc_image_count) vkDestroySemaphore(device, render_finish_sem[i], NULL);
    ito(sc_image_count) vkDestroyFence(device, frame_fences[i], NULL);
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroyInstance(instance, NULL);
    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  u32 find_mem_chunk(u32 prop_flags, u32 memory_type_bits, u32 alignment, u32 size) {
    (void)alignment;
    ito(mem_chunks.size) { // look for a suitable memory chunk
      Mem_Chunk &chunk = mem_chunks[i];
      if ((chunk.prop_flags & prop_flags) == prop_flags &&
          (chunk.memory_type_bits & memory_type_bits) == memory_type_bits) {
        if (chunk.has_space(size)) {
          return i;
        }
      }
    }
    // if failed create a new one
    Mem_Chunk new_chunk;
    u32       num_pages = 1 << 13;
    if (num_pages * Mem_Chunk::PAGE_SIZE < size) {
      num_pages = (size + Mem_Chunk::PAGE_SIZE - 1) / Mem_Chunk::PAGE_SIZE;
    }
    new_chunk.init(device, num_pages, find_mem_type(memory_type_bits, prop_flags), prop_flags,
                   memory_type_bits);

    ASSERT_DEBUG(new_chunk.has_space(size));
    mem_chunks.push(new_chunk);
    return mem_chunks.size - 1;
  }

  u32 find_mem_type(u32 type, VkMemoryPropertyFlags prop_flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physdevice, &props);
    ito(props.memoryTypeCount) {
      if (type & (1 << i) && (props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) {
        return i;
      }
    }
    TRAP;
  }

  VkDeviceMemory alloc_memory(u32 property_flags, VkMemoryRequirements reqs) {
    VkMemoryAllocateInfo info;
    MEMZERO(info);
    info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize  = reqs.size;
    info.memoryTypeIndex = find_mem_type(reqs.memoryTypeBits, property_flags);
    VkDeviceMemory mem;
    VK_ASSERT_OK(vkAllocateMemory(device, &info, nullptr, &mem));
    return mem;
  }

  Pair<VkBuffer, VkDeviceMemory> create_transient_buffer(u32 size) {
    VkBuffer           buf;
    VkBufferCreateInfo cinfo;
    MEMZERO(cinfo);
    cinfo.sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    cinfo.pQueueFamilyIndices   = &graphics_queue_id;
    cinfo.queueFamilyIndexCount = 1;
    cinfo.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
    cinfo.size                  = size;
    cinfo.usage                 = 0;
    cinfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK_ASSERT_OK(vkCreateBuffer(device, &cinfo, NULL, &buf));
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buf, &reqs);
    VkDeviceMemory mem = alloc_memory(
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, reqs);
    VK_ASSERT_OK(vkBindBufferMemory(device, buf, mem, 0));
    return {buf, mem};
  }

  Resource_ID create_buffer(rd::Buffer info, void const *initial_data) {
    u32 prop_flags = 0;
    if (info.mem_bits & (i32)rd::Memory_Bits::MAPPABLE) {
      prop_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    if (info.mem_bits & (i32)rd::Memory_Bits::DEVICE) {
      prop_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    VkBuffer           buf;
    VkBufferCreateInfo cinfo;
    {
      MEMZERO(cinfo);
      cinfo.sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      cinfo.pQueueFamilyIndices   = &graphics_queue_id;
      cinfo.queueFamilyIndexCount = 1;
      cinfo.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
      cinfo.size                  = info.size;
      cinfo.usage                 = 0;
      cinfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      if (info.mem_bits & (i32)rd::Memory_Bits::MAPPABLE) {
        cinfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      }
      if (info.usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER) {
        cinfo.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      }
      if (info.usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER) {
        cinfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      }
      if (info.usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_UAV) {
        cinfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      }
      if (info.usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER) {
        cinfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      }
      VK_ASSERT_OK(vkCreateBuffer(device, &cinfo, NULL, &buf));
    }
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buf, &reqs);
    Buffer new_buf;
    new_buf.buffer       = buf;
    new_buf.access_flags = 0;
    new_buf.create_info  = cinfo;
    new_buf.ref_cnt      = 1;

    u32 chunk_index = find_mem_chunk(prop_flags, reqs.memoryTypeBits, reqs.alignment, reqs.size);
    new_buf.mem_chunk_id = ID{chunk_index + 1};
    Mem_Chunk &chunk     = mem_chunks[chunk_index];
    new_buf.mem_offset   = chunk.alloc(reqs.alignment, reqs.size);

    vkBindBufferMemory(device, new_buf.buffer, chunk.mem, new_buf.mem_offset);
    if (initial_data != NULL) {
      ASSERT_DEBUG(info.mem_bits & (i32)rd::Memory_Bits::MAPPABLE);
      void *data = NULL;
      VK_ASSERT_OK(
          vkMapMemory(device, chunk.mem, new_buf.mem_offset, new_buf.create_info.size, 0, &data));
      memcpy(data, initial_data, new_buf.create_info.size);
      vkUnmapMemory(device, chunk.mem);
    }

    return {buffers.push(new_buf), (i32)rd::Type::Buffer};
  }

  void *map_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Buffer);
    Buffer &   buf   = buffers[res_id.id];
    Mem_Chunk &chunk = mem_chunks[buf.mem_chunk_id.index()];
    void *     data  = NULL;
    VK_ASSERT_OK(vkMapMemory(device, chunk.mem, buf.mem_offset, buf.create_info.size, 0, &data));
    return data;
  }

  void unmap_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Buffer);
    Buffer &   buf   = buffers[res_id.id];
    Mem_Chunk &chunk = mem_chunks[buf.mem_chunk_id.index()];
    vkUnmapMemory(device, chunk.mem);
  }

  VkShaderModule compile_spirv(size_t len, u32 *bytecode) {
    VkShaderModuleCreateInfo info;
    MEMZERO(info);
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = len;
    info.flags    = 0;
    info.pCode    = bytecode;
    VkShaderModule module;
    VK_ASSERT_OK(vkCreateShaderModule(device, &info, NULL, &module));
    return module;
  }

  void invalidate_cache() {}

  Resource_ID create_render_pass(string_ref name, u32 width, u32 height, string_ref *deps,
                                 u32 num_deps, string_ref *rts_names, RT_Info *rts, u32 num_rts,
                                 List *src) {
    if (named_render_passes.contains(name)) {
      ID           pass_id    = named_render_passes.get(name);
      Render_Pass &pass       = render_passes[pass_id];
      bool         invalidate = false;
      if (                              //
          pass.width != width ||        //
          pass.height != height ||      //
          pass.deps.size != num_deps || //
          pass.rts.size != num_rts) {
        invalidate = true;
      }

      if (invalidate) {
        ito(pass.rts.size) {
          Resource_ID res_id = named_resources.get(pass.rts[i]);
          named_resources.remove(pass.rts[i]);
          ASSERT_DEBUG(res_id.type == (i32)rd::Type::RT);
          this->rts.remove(res_id.id, 3);
        }
        render_passes.remove(pass_id, 3);
        named_render_passes.remove(name);
        invalidate_cache();
        goto make_new;
      } else
        return {pass_id, (i32)rd::Type::RenderPass};
    }

  make_new:
    Render_Pass pass;
    pass.init();
    pass.name   = relocate_cstr(name);
    pass.width  = width;
    pass.height = height;
    pass.src    = src;
    ito(num_deps) {
      Pass_Input input;
      if (deps[i].ptr[0] == '~') {
        input.name    = relocate_cstr(deps[i].substr(1, deps[i].len - 1));
        input.history = true;
      } else {
        input.name    = relocate_cstr(deps[i]);
        input.history = false;
      }
      pass.deps.push(input);
    }
    ito(num_rts) {
      string_ref  rt_name = relocate_cstr(rts_names[i]);
      Resource_ID rt      = create_rt(rts[i], width, height);
      ASSERT_DEBUG(!named_resources.contains(rts_names[i]));
      named_resources.insert(rt_name, rt);
      if (rts[i].type == rd::RT_t::Depth) {
        ASSERT_DEBUG(pass.depth_target.is_null());
        pass.depth_target = rt;
      }
      pass.rts.push(rt_name);
    }
    SmallArray<VkAttachmentDescription, 6> attachments;
    SmallArray<VkAttachmentReference, 6>   refs;
    attachments.init();
    refs.init();
    defer({
      attachments.release();
      refs.release();
    });
    u32 depth_attachment_id = 0;
    ito(num_rts) {
      VkAttachmentDescription attachment;
      MEMZERO(attachment);
      if (rts[i].type == rd::RT_t::Color) {
        attachment.format         = rts[i].format;
        attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_attachment;
        MEMZERO(color_attachment);
        color_attachment.attachment = i;
        color_attachment.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        refs.push(color_attachment);
      } else {
        attachment.format         = rts[i].format;
        attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_id       = i;
      }
      attachments.push(attachment);
    }
    VkRenderPassCreateInfo cinfo;
    MEMZERO(cinfo);
    cinfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    cinfo.attachmentCount = attachments.size;
    cinfo.pAttachments    = &attachments[0];

    VkSubpassDescription  subpass;
    VkAttachmentReference depth_attachment;
    MEMZERO(subpass);
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = refs.size;
    subpass.pColorAttachments    = &refs[0];
    if (!pass.depth_target.is_null()) {
      MEMZERO(depth_attachment);
      depth_attachment.attachment     = depth_attachment_id;
      depth_attachment.layout         = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      subpass.pDepthStencilAttachment = &depth_attachment;
    }
    cinfo.pSubpasses   = &subpass;
    cinfo.subpassCount = 1;

    VkSubpassDependency dependency;
    MEMZERO(dependency);
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0;
    dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    cinfo.pDependencies   = &dependency;
    cinfo.dependencyCount = 1;

    VK_ASSERT_OK(vkCreateRenderPass(device, &cinfo, NULL, &pass.pass));

    {
      SmallArray<VkImageView, 8> views;
      views.init();
      defer(views.release());
      ito(pass.rts.size) {
        Resource_ID res_id = named_resources.get(pass.rts[i]);
        ASSERT_ALWAYS(res_id.type == (i32)rd::Type::RT);
        RT &       rt   = this->rts[res_id.id];
        ImageView &view = image_views[rt.view_id];
        views.push(view.view);
      }
      VkFramebufferCreateInfo info;
      MEMZERO(info);
      info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      info.attachmentCount = views.size;
      info.width           = width;
      info.height          = height;
      info.layers          = 1;
      info.pAttachments    = &views[0];
      info.renderPass      = pass.pass;
      VK_ASSERT_OK(vkCreateFramebuffer(device, &info, NULL, &pass.fb));
    }
    ID pass_id = render_passes.push(pass);
    named_render_passes.insert(pass.name, pass_id);
    return {pass_id, (i32)rd::Type::RenderPass};
  }

  Resource_ID create_rt(RT_Info desc, u32 width, u32 height) {
    Resource_ID image =
        create_image(width, height, 1, 1, 1, desc.format,
                     desc.type == rd::RT_t::Color ? (i32)rd::Image_Usage_Bits::USAGE_RT
                                                  : (i32)rd::Image_Usage_Bits::USAGE_DT,
                     (i32)rd::Memory_Bits::DEVICE);
    Resource_ID image_view = create_image_view(image, 0, 1, 0, 1);
    RT          rt;
    rt.img_id  = image.id;
    rt.view_id = image_view.id;
    return {rts.push(rt), (i32)rd::Type::RT};
  }

  Resource_ID create_image_view(Resource_ID res_id, u32 base_level, u32 levels, u32 base_layer,
                                u32 layers) {
    ASSERT_ALWAYS(res_id.type == (i32)rd::Type::Image);
    Image &               img = images[res_id.id];
    ImageView             img_view;
    VkImageViewCreateInfo cinfo;
    MEMZERO(cinfo);
    cinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    VkComponentMapping cm;
    cm.r                                  = VK_COMPONENT_SWIZZLE_R;
    cm.g                                  = VK_COMPONENT_SWIZZLE_G;
    cm.b                                  = VK_COMPONENT_SWIZZLE_B;
    cm.a                                  = VK_COMPONENT_SWIZZLE_A;
    cinfo.components                      = cm;
    cinfo.format                          = img.create_info.format;
    cinfo.image                           = img.image;
    cinfo.subresourceRange.aspectMask     = img.aspect;
    cinfo.subresourceRange.baseArrayLayer = base_layer;
    cinfo.subresourceRange.baseMipLevel   = base_level;
    cinfo.subresourceRange.layerCount     = layers;
    cinfo.subresourceRange.levelCount     = levels;
    cinfo.viewType                        = img.create_info.extent.depth == 1
                         ? (img.create_info.extent.height == 1 ? //
                                (img.create_info.arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_1D
                                                                  : VK_IMAGE_VIEW_TYPE_1D_ARRAY)
                                                               : //
                                (img.create_info.arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_2D
                                                                  : VK_IMAGE_VIEW_TYPE_2D_ARRAY))
                         : VK_IMAGE_VIEW_TYPE_3D;
    VK_ASSERT_OK(vkCreateImageView(device, &cinfo, NULL, &img_view.view));
    img_view.img_id = res_id.id;
    img.add_reference();
    img_view.create_info = cinfo;
    return {image_views.push(img_view), (i32)rd::Type::ImageView};
  }

  Resource_ID create_image(u32 width, u32 height, u32 depth, u32 layers, u32 levels,
                           VkFormat format, u32 usage_flags, u32 mem_flags) {
    u32 prop_flags = 0;
    if (mem_flags & (i32)rd::Memory_Bits::MAPPABLE) {
      prop_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    if (mem_flags & (i32)rd::Memory_Bits::DEVICE) {
      prop_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    VkImage           image;
    VkImageCreateInfo cinfo;
    {
      MEMZERO(cinfo);
      cinfo.sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      cinfo.pQueueFamilyIndices   = &graphics_queue_id;
      cinfo.queueFamilyIndexCount = 1;
      cinfo.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
      cinfo.usage                 = 0;
      cinfo.extent                = VkExtent3D{width, height, depth};
      cinfo.arrayLayers           = layers;
      cinfo.mipLevels             = levels;
      cinfo.samples               = VK_SAMPLE_COUNT_1_BIT;
      cinfo.imageType =
          depth == 1 ? (height == 1 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D) : VK_IMAGE_TYPE_3D;
      cinfo.format        = format;
      cinfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      cinfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
      cinfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      if (mem_flags & (i32)rd::Memory_Bits::MAPPABLE) {
        cinfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      }
      if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_RT) {
        cinfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      }
      if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_DT) {
        cinfo.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      }
      if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_SAMPLED) {
        cinfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
      }
      if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_UAV) {
        cinfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      }
      VK_ASSERT_OK(vkCreateImage(device, &cinfo, NULL, &image));
    }
    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(device, image, &reqs);
    Image new_image;
    new_image.image        = image;
    new_image.access_flags = 0;
    new_image.create_info  = cinfo;
    new_image.layout       = VK_IMAGE_LAYOUT_UNDEFINED;
    new_image.ref_cnt      = 1;
    if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_DT)
      new_image.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else
      new_image.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    u32 chunk_index = find_mem_chunk(prop_flags, reqs.memoryTypeBits, reqs.alignment, reqs.size);
    new_image.mem_chunk_id = ID{chunk_index + 1};
    Mem_Chunk &chunk       = mem_chunks[chunk_index];
    new_image.mem_offset   = chunk.alloc(reqs.alignment, reqs.size);
    vkBindImageMemory(device, new_image.image, chunk.mem, new_image.mem_offset);
    return {images.push(new_image), (i32)rd::Type::Image};
  }

  // Resource_ID alloc_shader(size_t len, u32 *bytecode) {
  //   Shader sh;
  //   sh.module = compile_spirv(len, bytecode);
  //   return {shaders.push(sh), (i32)rd::Type::Shader};
  // }

  Resource_ID alloc_shader(List *root) {
    Shader sh;
    sh.root = root;
    sh.hash = hash_of(root->get_umbrella_string());
    return {shaders.push(sh), (i32)rd::Type::Shader};
  }

  void release_resource(Resource_ID res_id) {
    if (res_id.type == (i32)rd::Type::Buffer) {
      buffers[res_id.id].rem_reference();
      //      buffers.remove(res_id.id, 3);
    } else if (res_id.type == (i32)rd::Type::Shader) {
      //      shaders.remove(res_id.id, 3);
      shaders[res_id.id].rem_reference();
    } else {
      TRAP;
    }
  }

  void VS_bind_shader(Resource_ID res_id) {
    if (res_id.is_null()) {
      graphics_state.vs      = ID{0};
      graphics_state.vs_hash = 0;
      return;
    }
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Shader);
    Shader &sh             = shaders[res_id.id];
    graphics_state.vs      = res_id.id;
    graphics_state.vs_hash = sh.hash;
  }

  void PS_bind_shader(Resource_ID res_id) {
    if (res_id.is_null()) {
      graphics_state.ps      = ID{0};
      graphics_state.ps_hash = 0;
      return;
    }
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Shader);
    Shader &sh             = shaders[res_id.id];
    graphics_state.ps      = res_id.id;
    graphics_state.ps_hash = sh.hash;
  }

  void IA_set_topology(VkPrimitiveTopology topo) { graphics_state.topology = topo; }

  void IA_bind_index_buffer(Resource_ID res_id, u32 offset, VkFormat format) {
    if (res_id.is_null()) {
      return;
    }
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Buffer);
    Buffer &    buf = buffers[res_id.id];
    VkIndexType type;
    switch (format) {
    case VK_FORMAT_R32_UINT: type = VK_INDEX_TYPE_UINT32; break;
    case VK_FORMAT_R16_UINT: type = VK_INDEX_TYPE_UINT16; break;
    default: TRAP;
    }
    vkCmdBindIndexBuffer(cmd_buffers[cmd_index], buf.buffer, (VkDeviceSize)offset, type);
  }

  void IA_bind_vertex_buffer(Resource_ID res_id, u32 binding, u32 offset, u32 stride,
                             VkVertexInputRate rate) {
    if (res_id.is_null()) {
      return;
    }
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Buffer);
    Buffer &     buf     = buffers[res_id.id];
    VkDeviceSize doffset = (VkDeviceSize)offset;
    vkCmdBindVertexBuffers(cmd_buffers[cmd_index], binding, 1, &buf.buffer, &doffset);
    if (graphics_state.num_bindings <= binding) {
      graphics_state.num_bindings = binding + 1;
    }
    graphics_state.bindings[binding].binding   = binding;
    graphics_state.bindings[binding].inputRate = rate;
    graphics_state.bindings[binding].stride    = stride;
  }

  void IA_bind_attribute(u32 location, u32 binding, u32 offset, VkFormat format) {
    if (graphics_state.num_attributes <= location) {
      graphics_state.num_attributes = location + 1;
    }
    graphics_state.attributes[location].binding  = binding;
    graphics_state.attributes[location].format   = format;
    graphics_state.attributes[location].location = location;
    graphics_state.attributes[location].offset   = offset;
  }

  void IA_set_cull_mode(VkFrontFace ff, VkCullModeFlags cm) {
    graphics_state.front_face = ff;
    graphics_state.cull_mode  = cm;
  }

  void bind_graphics_pipeline() {
    if (!pipeline_cache.contains(graphics_state)) {
      Graphics_Pipeline_Wrapper gw;
      ASSERT_DEBUG(!graphics_state.ps.is_null());
      ASSERT_DEBUG(!graphics_state.vs.is_null());
      Shader &ps = shaders[graphics_state.ps];
      Shader &vs = shaders[graphics_state.vs];
      ASSERT_DEBUG(!cur_pass.is_null());
      Render_Pass &pass = render_passes[cur_pass];
      gw.init(device, pass, vs, ps, graphics_state);
    }
  }

  void draw_indexed_instanced(u32 index_count, u32 instance_count, u32 start_index,
                              i32 start_vertex, u32 start_instance) {
    bind_graphics_pipeline();
    vkCmdDrawIndexed(cmd_buffers[cmd_index], index_count, instance_count, start_index, start_vertex,
                     start_instance);
  }

  //  u32 shader_get_input_location(Resource_ID res_id, string_ref name) {
  //    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Shader);
  //    Shader &sh = shaders[res_id.id];
  //    ito(sh.attributes.size) {
  //      if (sh.attributes[i].name == name) return sh.attributes[i].location;
  //    }
  //    TRAP;
  //  }

  void release_swapchain() {
    if (swapchain != VK_NULL_HANDLE) {
      vkDestroySwapchainKHR(device, swapchain, NULL);
    }
    ito(sc_image_count) {
      if (sc_framebuffers[i] != VK_NULL_HANDLE)
        vkDestroyFramebuffer(device, sc_framebuffers[i], NULL);
      if (sc_image_views[i] != VK_NULL_HANDLE) vkDestroyImageView(device, sc_image_views[i], NULL);
    }
    if (sc_render_pass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(device, sc_render_pass, NULL);
    }
  }

  void update_swapchain() {
    SDL_SetWindowResizable(window, SDL_FALSE);
    defer(SDL_SetWindowResizable(window, SDL_TRUE));
    vkDeviceWaitIdle(device);
    release_swapchain();
    u32                format_count = 0;
    VkSurfaceFormatKHR formats[0x100];
    vkGetPhysicalDeviceSurfaceFormatsKHR(physdevice, surface, &format_count, 0);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physdevice, surface, &format_count, formats);
    VkSurfaceFormatKHR format_of_choice;
    format_of_choice.format = VK_FORMAT_UNDEFINED;
    ito(format_count) {
      if (formats[i].format == VK_FORMAT_R8G8B8A8_SRGB ||  //
          formats[i].format == VK_FORMAT_B8G8R8A8_SRGB ||  //
          formats[i].format == VK_FORMAT_B8G8R8_SRGB ||    //
          formats[i].format == VK_FORMAT_R8G8B8_SRGB ||    //
          formats[i].format == VK_FORMAT_R8G8B8_UNORM ||   //
          formats[i].format == VK_FORMAT_R8G8B8A8_UNORM || //
          formats[i].format == VK_FORMAT_B8G8R8A8_UNORM || //
          formats[i].format == VK_FORMAT_B8G8R8_UNORM      //
      ) {
        format_of_choice = formats[i];
        break;
      }
    }
    ASSERT_ALWAYS(format_of_choice.format != VK_FORMAT_UNDEFINED);
    sc_format = format_of_choice;

    uint32_t num_present_modes = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physdevice, surface, &num_present_modes, NULL);
    VkPresentModeKHR present_modes[0x100];
    vkGetPhysicalDeviceSurfacePresentModesKHR(physdevice, surface, &num_present_modes,
                                              present_modes);
    VkPresentModeKHR present_mode_of_choice = VK_PRESENT_MODE_FIFO_KHR; // always supported.
    ito(num_present_modes) {
      if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) { // prefer mailbox
        present_mode_of_choice = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
      }
    }
    //    usleep(100000);
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdevice, surface, &surface_capabilities);

    sc_extent        = surface_capabilities.currentExtent;
    sc_extent.width  = CLAMP(sc_extent.width, surface_capabilities.minImageExtent.width,
                            surface_capabilities.maxImageExtent.width);
    sc_extent.height = CLAMP(sc_extent.height, surface_capabilities.minImageExtent.height,
                             surface_capabilities.maxImageExtent.height);

    VkSwapchainCreateInfoKHR sc_create_info;
    MEMZERO(sc_create_info);
    sc_create_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sc_create_info.surface          = surface;
    sc_create_info.minImageCount    = CLAMP(3, surface_capabilities.minImageCount, 0x10);
    sc_create_info.imageFormat      = format_of_choice.format;
    sc_create_info.imageColorSpace  = format_of_choice.colorSpace;
    sc_create_info.imageExtent      = sc_extent;
    sc_create_info.imageArrayLayers = 1;
    sc_create_info.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sc_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc_create_info.preTransform     = surface_capabilities.currentTransform;
    sc_create_info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc_create_info.presentMode      = present_mode_of_choice;
    sc_create_info.clipped          = VK_TRUE;
    sc_image_count                  = 0;
    VK_ASSERT_OK(vkCreateSwapchainKHR(device, &sc_create_info, 0, &swapchain));
    vkGetSwapchainImagesKHR(device, swapchain, &sc_image_count, NULL);
    vkGetSwapchainImagesKHR(device, swapchain, &sc_image_count, sc_images);
    ito(sc_image_count) {
      VkImageViewCreateInfo view_ci;
      MEMZERO(view_ci);
      view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_ci.image = sc_images[i];
      MEMZERO(view_ci.components);
      view_ci.components.r = VK_COMPONENT_SWIZZLE_R;
      view_ci.components.g = VK_COMPONENT_SWIZZLE_G;
      view_ci.components.b = VK_COMPONENT_SWIZZLE_B;
      view_ci.components.a = VK_COMPONENT_SWIZZLE_A;
      view_ci.format       = sc_format.format;
      MEMZERO(view_ci.subresourceRange);
      view_ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      view_ci.subresourceRange.baseMipLevel   = 0;
      view_ci.subresourceRange.levelCount     = 1;
      view_ci.subresourceRange.baseArrayLayer = 0;
      view_ci.subresourceRange.layerCount     = 1;
      view_ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
      VK_ASSERT_OK(vkCreateImageView(device, &view_ci, NULL, &sc_image_views[i]));
      sc_image_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    {
      VkAttachmentDescription attachment;
      MEMZERO(attachment);
      attachment.format         = sc_format.format;
      attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
      attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
      attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
      attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      attachment.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      VkAttachmentReference color_attachment;
      MEMZERO(color_attachment);
      color_attachment.attachment = 0;
      color_attachment.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      VkSubpassDescription subpass;
      MEMZERO(subpass);
      subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments    = &color_attachment;
      VkSubpassDependency dependency;
      MEMZERO(dependency);
      dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass    = 0;
      dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      VkRenderPassCreateInfo info;
      MEMZERO(info);
      info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      info.attachmentCount = 1;
      info.pAttachments    = &attachment;
      info.subpassCount    = 1;
      info.pSubpasses      = &subpass;
      info.dependencyCount = 1;
      info.pDependencies   = &dependency;
      VK_ASSERT_OK(vkCreateRenderPass(device, &info, NULL, &sc_render_pass));
    }
    ito(sc_image_count) {
      VkFramebufferCreateInfo info;
      MEMZERO(info);
      info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      info.attachmentCount = 1;
      info.width           = sc_extent.width;
      info.height          = sc_extent.height;
      info.layers          = 1;
      info.pAttachments    = &sc_image_views[i];
      info.renderPass      = sc_render_pass;
      VK_ASSERT_OK(vkCreateFramebuffer(device, &info, NULL, &sc_framebuffers[i]));
    }
  }

  void init() {
    init_ds();
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    SDL_Window *window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    TMP_STORAGE_SCOPE;

    u32 num_instance_extensions;
    ASSERT_ALWAYS(SDL_Vulkan_GetInstanceExtensions(window, &num_instance_extensions, nullptr));
    const char **instance_extensions =
        (char const **)tl_alloc_tmp((num_instance_extensions + 1) * sizeof(char *));
    ASSERT_ALWAYS(
        SDL_Vulkan_GetInstanceExtensions(window, &num_instance_extensions, instance_extensions));
    instance_extensions[num_instance_extensions++] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;

    VkApplicationInfo app_info;
    MEMZERO(app_info);
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion         = VK_API_VERSION_1_2;
    app_info.applicationVersion = 1;
    app_info.pApplicationName   = "Vulkii";
    app_info.pEngineName        = "Vulkii";

    const char *layerNames[] = {//
                                // "VK_LAYER_LUNARG_standard_validation" // [Deprecated]
                                "VK_LAYER_KHRONOS_validation", NULL};

    VkInstanceCreateInfo info;
    MEMZERO(info);
    info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo        = &app_info;
    info.enabledLayerCount       = ARRAY_SIZE(layerNames) - 1;
    info.ppEnabledLayerNames     = layerNames;
    info.enabledExtensionCount   = num_instance_extensions;
    info.ppEnabledExtensionNames = instance_extensions;

    VK_ASSERT_OK(vkCreateInstance(&info, nullptr, &instance));

    if (!SDL_Vulkan_CreateSurface(window, instance, &surface)) {
      TRAP;
    }
    const u32               MAX_COUNT = 0x100;
    u32                     physdevice_count;
    VkPhysicalDevice        physdevice_handles[MAX_COUNT];
    VkQueueFamilyProperties queue_family_properties[MAX_COUNT];
    //  VkQueueFamilyProperties2    queue_family_properties2[MAX_COUNT];
    //  VkQueueFamilyProperties2KHR queue_family_properties2KHR[MAX_COUNT];

    vkEnumeratePhysicalDevices(instance, &physdevice_count, 0);
    vkEnumeratePhysicalDevices(instance, &physdevice_count, physdevice_handles);

    VkPhysicalDevice graphics_device_id = NULL;

    ito(physdevice_count) {
      {
        u32 num_queue_family_properties = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physdevice_handles[i],
                                                 &num_queue_family_properties, NULL);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physdevice_handles[i], &num_queue_family_properties, queue_family_properties);

        jto(num_queue_family_properties) {

          VkBool32 sup = VK_FALSE;
          vkGetPhysicalDeviceSurfaceSupportKHR(physdevice_handles[i], j, surface, &sup);

          if (sup && (queue_family_properties[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            graphics_queue_id  = j;
            graphics_device_id = physdevice_handles[i];
          }
          if (sup && (queue_family_properties[j].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            compute_queue_id = j;
          }
          if (sup && (queue_family_properties[j].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
            transfer_queue_id = j;
          }
        }
      }
      //    {
      //      u32 num_queue_family_properties = 0;
      //      vkGetPhysicalDeviceQueueFamilyProperties2(physdevice_handles[i],
      //      &num_queue_family_properties,
      //                                                NULL);
      //      vkGetPhysicalDeviceQueueFamilyProperties2(physdevice_handles[i],
      //      &num_queue_family_properties,
      //                                                queue_family_properties2);
      //    }
      //    {
      //      u32 num_queue_family_properties = 0;
      //      vkGetPhysicalDeviceQueueFamilyProperties2KHR(physdevice_handles[i],
      //                                                   &num_queue_family_properties, NULL);
      //      vkGetPhysicalDeviceQueueFamilyProperties2KHR(
      //          physdevice_handles[i], &num_queue_family_properties,
      //          queue_family_properties2KHR);
      //    }
    }
    char const *       device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo deviceCreateInfo;
    MEMZERO(deviceCreateInfo);
    deviceCreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount    = 1;
    deviceCreateInfo.pQueueCreateInfos       = 0;
    deviceCreateInfo.enabledLayerCount       = 0;
    deviceCreateInfo.ppEnabledLayerNames     = 0;
    deviceCreateInfo.enabledExtensionCount   = ARRAY_SIZE(device_extensions);
    deviceCreateInfo.ppEnabledExtensionNames = device_extensions;
    deviceCreateInfo.pEnabledFeatures        = 0;
    float                   priority         = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info;
    MEMZERO(queue_create_info);
    queue_create_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = graphics_queue_id;
    queue_create_info.queueCount       = 1;
    queue_create_info.pQueuePriorities = &priority;
    deviceCreateInfo.pQueueCreateInfos = &queue_create_info;
    VK_ASSERT_OK(vkCreateDevice(graphics_device_id, &deviceCreateInfo, NULL, &device));
    vkGetDeviceQueue(device, graphics_queue_id, 0, &queue);
    ASSERT_ALWAYS(queue != VK_NULL_HANDLE);
    physdevice = graphics_device_id;
    update_swapchain();
    {
      VkCommandPoolCreateInfo info;
      MEMZERO(info);
      info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      info.queueFamilyIndex = graphics_queue_id;

      vkCreateCommandPool(device, &info, 0, &cmd_pool);
    }
    {
      VkCommandBufferAllocateInfo info;
      MEMZERO(info);
      info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      info.commandPool        = cmd_pool;
      info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      info.commandBufferCount = sc_image_count;

      vkAllocateCommandBuffers(device, &info, cmd_buffers);
    }
    {
      VkSemaphoreCreateInfo info;
      MEMZERO(info);
      info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      info.pNext = NULL;
      info.flags = 0;
      ito(sc_image_count) vkCreateSemaphore(device, &info, 0, &sc_free_sem[i]);
      ito(sc_image_count) vkCreateSemaphore(device, &info, 0, &render_finish_sem[i]);
    }
    {
      VkFenceCreateInfo info;
      MEMZERO(info);
      info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      ito(sc_image_count) vkCreateFence(device, &info, 0, &frame_fences[i]);
    }
  }

  void update_surface_size() {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdevice, surface, &surface_capabilities);
    window_width  = surface_capabilities.currentExtent.width;
    window_height = surface_capabilities.currentExtent.height;
  }
  void start_frame() {
    graphics_state.reset();
    buffers.tick();
    images.tick();
    shaders.tick();
    buffer_views.tick();
    image_views.tick();
    render_passes.tick();
    rts.tick();
    pipelines.tick();
    images.for_each([this](Image &image) {
      if (!image.is_referenced()) images.remove(image.get_id(), 3);
    });
    buffers.for_each([this](Buffer &buf) {
      if (!buf.is_referenced()) buffers.remove(buf.get_id(), 3);
    });
  restart:
    update_surface_size();
    if (window_width != (i32)sc_extent.width || window_height != (i32)sc_extent.height) {
      update_swapchain();
    }

    cmd_index         = (frame_id++) % sc_image_count;
    VkResult wait_res = vkWaitForFences(device, 1, &frame_fences[cmd_index], VK_TRUE, 1000);
    if (wait_res == VK_TIMEOUT) {
      goto restart;
    }
    vkResetFences(device, 1, &frame_fences[cmd_index]);

    VkResult acquire_res = vkAcquireNextImageKHR(
        device, swapchain, UINT64_MAX, sc_free_sem[cmd_index], VK_NULL_HANDLE, &image_index);

    if (acquire_res == VK_ERROR_OUT_OF_DATE_KHR || acquire_res == VK_SUBOPTIMAL_KHR) {
      update_swapchain();
      goto restart;
    } else if (acquire_res != VK_SUCCESS) {
      TRAP;
    }

    VkCommandBufferBeginInfo begin_info;
    MEMZERO(begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd_buffers[cmd_index], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    vkBeginCommandBuffer(cmd_buffers[cmd_index], &begin_info);
    VkImageSubresourceRange srange;
    MEMZERO(srange);
    srange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    srange.baseMipLevel   = 0;
    srange.levelCount     = VK_REMAINING_MIP_LEVELS;
    srange.baseArrayLayer = 0;
    srange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
    {
      VkImageMemoryBarrier bar;
      MEMZERO(bar);
      bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bar.srcAccessMask       = 0;
      bar.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      bar.oldLayout           = sc_image_layout[image_index];
      bar.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      bar.srcQueueFamilyIndex = graphics_queue_id;
      bar.dstQueueFamilyIndex = graphics_queue_id;
      bar.image               = sc_images[image_index];
      bar.subresourceRange    = srange;
      vkCmdPipelineBarrier(cmd_buffers[cmd_index], VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
    }
    VkClearColorValue clear_color = {{1.0f, 0.0f, 0.0f, 1.0f}};
    vkCmdClearColorImage(cmd_buffers[cmd_index], sc_images[image_index],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &srange);
    {
      VkImageMemoryBarrier bar;
      MEMZERO(bar);
      bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bar.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      bar.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
      bar.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      bar.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      bar.srcQueueFamilyIndex = graphics_queue_id;
      bar.dstQueueFamilyIndex = graphics_queue_id;
      bar.image               = sc_images[image_index];
      bar.subresourceRange    = srange;
      vkCmdPipelineBarrier(cmd_buffers[cmd_index], VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
    }
    sc_image_layout[image_index] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  }
  void end_frame() {
    vkEndCommandBuffer(cmd_buffers[cmd_index]);
    VkPipelineStageFlags stage_flags[]{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo         submit_info;
    MEMZERO(submit_info);
    submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount   = 1;
    submit_info.pWaitSemaphores      = &sc_free_sem[cmd_index];
    submit_info.pWaitDstStageMask    = stage_flags;
    submit_info.commandBufferCount   = 1;
    submit_info.pCommandBuffers      = &cmd_buffers[cmd_index];
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores    = &render_finish_sem[cmd_index];
    vkQueueSubmit(queue, 1, &submit_info, frame_fences[cmd_index]);
    VkPresentInfoKHR present_info;
    MEMZERO(present_info);
    present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores    = &render_finish_sem[cmd_index];
    present_info.swapchainCount     = 1;
    present_info.pSwapchains        = &swapchain;
    present_info.pImageIndices      = &image_index;
    vkQueuePresentKHR(queue, &present_info);
  }
};

// typedef void (*Loop_Callback_t)(rd::Pass_Mng *);

enum class Render_Value_t {
  UNKNOWN = 0,
  RESOURCE_ID,
  FLAGS,
  ARRAY,
  //  SHADER_SOURCE,
};

struct Render_Value {
  struct Array {
    void *ptr;
    u32   size;
  };
  //  struct Shader_Source {
  //    shaderc_shader_kind kind;
  //    string_ref          text;
  //  };
  union {
    Resource_ID res_id;
    Array       arr;
    //    Shader_Source shsrc;
  };
};

struct Renderign_Evaluator final : public IEvaluator {
  Window             wnd;
  Pool<Render_Value> rd_values;
  Pool<>             tmp_values;
  // scratch state
  struct Scratch_State {
    string_ref        name;
    i32               index_count;
    i32               instance_count;
    i32               start_index;
    i32               start_vertex;
    i32               start_innstance;
    i32               offset;
    i32               binding;
    i32               location;
    i32               stride;
    VkVertexInputRate rate;
    VkFormat          format;
    VkFrontFace       front_face;
    VkCullModeFlags   cull_mode;
    rd::RT_t          rt_type;
    void              clear() { memset(this, 0, sizeof(*this)); }
    void              eval(List *l) {
      if (l->child != NULL) {
        eval(l->child);
      } else if (l->cmp_symbol("offset")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &offset));
      } else if (l->cmp_symbol("binding")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &binding));
      } else if (l->cmp_symbol("stride")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &stride));
      } else if (l->cmp_symbol("location")) {
        ASSERT_DEBUG(parse_decimal_int(l->next->symbol.ptr, l->next->symbol.len, &location));
      } else if (l->cmp_symbol("format")) {
        format = parse_format(l->next->symbol);
      } else if (l->cmp_symbol("name")) {
        name = l->next->symbol;
      } else if (l->cmp_symbol("rate")) {
        if (l->next->cmp_symbol("PER_VERTEX")) {
          rate = VK_VERTEX_INPUT_RATE_VERTEX;
        } else if (l->next->cmp_symbol("PER_INSTANCE")) {
          rate = VK_VERTEX_INPUT_RATE_INSTANCE;
        } else {
          TRAP;
        }
      } else if (l->cmp_symbol("front_face")) {
        if (l->next->cmp_symbol("CCW")) {
          front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        } else if (l->next->cmp_symbol("CW")) {
          front_face = VK_FRONT_FACE_CLOCKWISE;
        } else {
          TRAP;
        }
      } else if (l->cmp_symbol("cull_mode")) {
        if (l->next->cmp_symbol("NONE")) {
          cull_mode = VK_CULL_MODE_NONE;
        } else if (l->next->cmp_symbol("FRONT")) {
          cull_mode = VK_CULL_MODE_FRONT_BIT;
        } else {
          TRAP;
        }
      } else if (l->cmp_symbol("rt_type")) {
        if (l->next->cmp_symbol("COLOR")) {
          rt_type = rd::RT_t::Color;
        } else if (l->next->cmp_symbol("DEPTH")) {
          rt_type = rd::RT_t::Depth;
        } else {
          TRAP;
        }
      }
    }
  } scratch;
  void init() {
    rd_values  = Pool<Render_Value>::create(1 << 10);
    tmp_values = Pool<>::create(1 << 10);
    wnd.init();
  }
  Renderign_Evaluator *create() {
    Renderign_Evaluator *out = new Renderign_Evaluator;
    out->init();
    return out;
  }
  void *alloc_tmp(u32 size) { return tmp_values.alloc(size); }
  void  enter_scope() {
    state->enter_scope();
    tmp_values.enter_scope();
    rd_values.enter_scope();
    string_storage.enter_scope();
  }
  void exit_scope() {
    state->exit_scope();
    tmp_values.exit_scope();
    rd_values.exit_scope();
    string_storage.exit_scope();
  }
  void release() override {
    wnd.release();
    tmp_values.release();
    rd_values.release();
    delete this;
  }
  void start_frame() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        exit(0);
      }
      switch (event.type) {
      case SDL_WINDOWEVENT:
        if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
        }
        break;
      }
    }
    wnd.start_frame();
  }
  Value *wrap_flags(u32 flags) {
    Value *new_val    = state->value_storage.alloc_zero(1);
    new_val->i        = (i32)flags;
    new_val->type     = (i32)Value::Value_t::ANY;
    new_val->any_type = (i32)Render_Value_t::FLAGS;
    return new_val;
  }
  Value *wrap_resource(Resource_ID res_id) {
    Render_Value *rval = rd_values.alloc_zero(1);
    rval->res_id       = res_id;
    Value *new_val     = state->value_storage.alloc_zero(1);
    new_val->type      = (i32)Value::Value_t::ANY;
    new_val->any_type  = (i32)Render_Value_t::RESOURCE_ID;
    new_val->any       = rval;
    return new_val;
  }
  Value *wrap_array(void *ptr, u32 size) {
    Render_Value *rval = rd_values.alloc_zero(1);
    rval->arr.ptr      = ptr;
    rval->arr.size     = size;
    Value *new_val     = state->value_storage.alloc_zero(1);
    new_val->type      = (i32)Value::Value_t::ANY;
    new_val->any_type  = (i32)Render_Value_t::ARRAY;
    new_val->any       = rval;
    return new_val;
  }
  //  Value *wrap_shader_source(string_ref text, shaderc_shader_kind kind) {
  //    Render_Value *rval = rd_values.alloc_zero(1);
  //    rval->shsrc.kind   = kind;
  //    rval->shsrc.text   = text;
  //    Value *new_val     = state->value_storage.alloc_zero(1);
  //    new_val->type      = (i32)Value::Value_t::ANY;
  //    new_val->any_type  = (i32)Render_Value_t::SHADER_SOURCE;
  //    new_val->any       = rval;
  //    return new_val;
  //  }
  void  end_frame() { wnd.end_frame(); }
  Match eval(List *l) override {
    if (l == NULL) return NULL;
    if (l->child != NULL) {
      return global_eval(l->child);
    } else if (l->cmp_symbol("start_frame")) {
      start_frame();
      return NULL;
    } else if (l->cmp_symbol("render-loop")) {
      while (true) {
        enter_scope();
        eval_args(l->next);
        exit_scope();
      }
      return NULL;
    } else if (l->cmp_symbol("end_frame")) {
      end_frame();
      return NULL;
    } else if (l->cmp_symbol("flags")) {
      List *cur   = l->next;
      u32   flags = 0;
      while (cur != NULL) {
        if (cur->cmp_symbol("Buffer_Usage_Bits::USAGE_TRANSIENT")) {
          flags |= (i32)rd::Buffer_Usage_Bits::USAGE_TRANSIENT;
        } else if (cur->cmp_symbol("Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER")) {
          flags |= (i32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
        } else if (cur->cmp_symbol("Buffer_Usage_Bits::USAGE_INDEX_BUFFER")) {
          flags |= (i32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
        } else if (cur->cmp_symbol("Buffer_Usage_Bits::USAGE_VERTEX_BUFFER")) {
          flags |= (i32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
        } else if (cur->cmp_symbol("Buffer_Usage_Bits::USAGE_UAV")) {
          flags |= (i32)rd::Buffer_Usage_Bits::USAGE_UAV;
        } else if (cur->cmp_symbol("Memory_Bits::MAPPABLE")) {
          flags |= (i32)rd::Memory_Bits::MAPPABLE;
        } else {
          ASSERT_DEBUG(false);
        }
        cur = cur->next;
      }
      return wrap_flags(flags);
    } else if (l->cmp_symbol("show_stats")) {
      wnd.buffers.dump();
      wnd.images.dump();
      wnd.shaders.dump();
      wnd.buffer_views.dump();
      wnd.image_views.dump();
      wnd.render_passes.dump();
      wnd.rts.dump();
      wnd.pipelines.dump();
      fprintf(stdout, "num mem chunks: %i\n", (i32)wnd.mem_chunks.size);
      ito(wnd.mem_chunks.size) wnd.mem_chunks[i].dump();
      return NULL;
    } else if (l->cmp_symbol("create_buffer")) {
      SmallArray<Value *, 2> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 3);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::FLAGS);
      ASSERT_EVAL(args[1]->type == (i32)Value::Value_t::ANY &&
                  args[1]->any_type == (i32)Render_Value_t::FLAGS);
      ASSERT_EVAL(args[2]->type == (i32)Value::Value_t::I32);
      rd::Buffer info;
      info.usage_bits    = args[0]->i;
      info.mem_bits      = args[1]->i;
      info.size          = args[2]->i;
      Resource_ID res_id = wnd.create_buffer(info, NULL);
      return wrap_resource(res_id);
    } else if (l->cmp_symbol("release_resource")) {
      SmallArray<Value *, 1> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)args[0]->any;
      wnd.release_resource(rval->res_id);
      return NULL;
    } else if (l->cmp_symbol("map_buffer")) {
      SmallArray<Value *, 2> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)args[0]->any;
      void *        ptr  = wnd.map_buffer(rval->res_id);
      return wrap_array(ptr, 0);
    } else if (l->cmp_symbol("unmap_buffer")) {
      SmallArray<Value *, 2> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)args[0]->any;
      wnd.unmap_buffer(rval->res_id);
      return NULL;
    } else if (l->cmp_symbol("array_f32")) {
      SmallArray<Value *, 16> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      u8 *data = (u8 *)alloc_tmp(4 * args.size);
      ito(args.size) {
        ASSERT_EVAL(args[i]->type == (i32)Value::Value_t::F32);
        memcpy(data + i * 4, &args[i]->f, 4);
      }
      return wrap_array(data, args.size * 4);
    } else if (l->cmp_symbol("array_i32")) {
      SmallArray<Value *, 16> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      u8 *data = (u8 *)alloc_tmp(4 * args.size);
      ito(args.size) {
        ASSERT_EVAL(args[i]->type == (i32)Value::Value_t::I32);
        memcpy(data + i * 4, &args[i]->f, 4);
      }
      return wrap_array(data, args.size * 4);
    } else if (l->cmp_symbol("memcpy")) {
      SmallArray<Value *, 4> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size == 2);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::ARRAY);
      ASSERT_EVAL(args[1]->type == (i32)Value::Value_t::ANY &&
                  args[1]->any_type == (i32)Render_Value_t::ARRAY);
      Render_Value *val_1 = (Render_Value *)args[0]->any;
      Render_Value *val_2 = (Render_Value *)args[1]->any;
      memcpy(val_1->arr.ptr, val_2->arr.ptr, val_2->arr.size);
      return NULL;
    } else if (l->cmp_symbol("create_shader")) {
      return wrap_resource(wnd.alloc_shader(l));
    } else if (l->cmp_symbol("VS_bind_shader")) {
      SmallArray<Value *, 1> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)args[0]->any;
      wnd.VS_bind_shader(rval->res_id);
      return NULL;
    } else if (l->cmp_symbol("PS_bind_shader")) {
      SmallArray<Value *, 1> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)args[0]->any;
      wnd.PS_bind_shader(rval->res_id);
      return NULL;
    } else if (l->cmp_symbol("IA_set_topology")) {
      SmallArray<Value *, 1> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size = 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::SYMBOL);
      VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      if (args[0]->str == stref_s("TRIANGLE_LIST")) {
        topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      } else if (args[0]->str == stref_s("LINE_LIST")) {
        topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      } else {
        TRAP;
      }
      wnd.IA_set_topology(topo);
      return NULL;
    } else if (l->cmp_symbol("IA_bind_index_buffer")) {
      Value *buf = global_eval(l->next).unwrap();
      ASSERT_EVAL(buf != NULL && buf->type == (i32)Value::Value_t::ANY &&
                  buf->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)buf->any;
      scratch.clear();
      List *cur = l->next->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      wnd.IA_bind_index_buffer(rval->res_id, scratch.offset, scratch.format);
      return NULL;
    } else if (l->cmp_symbol("IA_bind_vertex_buffer")) {
      Value *buf = global_eval(l->next).unwrap();
      ASSERT_EVAL(buf != NULL && buf->type == (i32)Value::Value_t::ANY &&
                  buf->any_type == (i32)Render_Value_t::RESOURCE_ID);
      Render_Value *rval = (Render_Value *)buf->any;
      scratch.clear();
      List *cur = l->next->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      wnd.IA_bind_vertex_buffer(rval->res_id, scratch.binding, scratch.offset, scratch.stride,
                                scratch.rate);
      return NULL;
    } else if (l->cmp_symbol("IA_bind_attribute")) {
      scratch.clear();
      List *cur = l->next->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      wnd.IA_bind_attribute(scratch.location, scratch.binding, scratch.offset, scratch.format);
      return NULL;
    } else if (l->cmp_symbol("IA_set_cull_mode")) {
      scratch.clear();
      List *cur = l->next->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      wnd.IA_set_cull_mode(scratch.front_face, scratch.cull_mode);
      return NULL;
    } else if (l->cmp_symbol("draw_indexed_instanced")) {
      scratch.clear();
      List *cur = l->next->next;
      while (cur != NULL) {
        scratch.eval(cur);
        cur = cur->next;
      }
      wnd.draw_indexed_instanced(scratch.index_count, scratch.instance_count, scratch.start_index,
                                 scratch.start_vertex, scratch.start_innstance);
      return NULL;
    } else if (l->cmp_symbol("clear_state")) {
      wnd.graphics_state.reset();
      return NULL;
    } else if (l->cmp_symbol("nil")) {
      return wrap_resource(Resource_ID{{0}, 0});
    } else if (l->cmp_symbol("start_render_pass")) {
      string_ref        name  = l->next->symbol;
      i32               width = 512, height = 512;
      Array<RT_Info>    rts;
      Array<string_ref> rts_names;
      Array<string_ref> deps_names;
      List *            src = NULL;
      rts.init();
      rts_names.init();
      deps_names.init();
      defer({
        rts.release();
        rts_names.release();
        deps_names.release();
      });
      List *cur = l->next->next;
      while (cur != NULL) {
        List *ch = cur->child;
        if (ch->cmp_symbol("width")) {
          if (ch->next->cmp_symbol("#")) {
            width = wnd.sc_extent.width;
          } else {
            ASSERT_DEBUG(parse_decimal_int(ch->next->symbol.ptr, ch->next->symbol.len, &width));
          }
        } else if (ch->cmp_symbol("height")) {
          if (ch->next->cmp_symbol("#")) {
            width = wnd.sc_extent.width;
          } else {
            ASSERT_DEBUG(parse_decimal_int(ch->next->symbol.ptr, ch->next->symbol.len, &height));
          }
        } else if (ch->cmp_symbol("depends")) {
          deps_names.push(ch->next->symbol);
        } else if (ch->cmp_symbol("add_render_target")) {
          scratch.clear();
          List *cur = ch->next;
          while (cur != NULL) {
            scratch.eval(cur);
            cur = cur->next;
          }
          RT_Info rt;
          rt.format = scratch.format;
          rt.type   = scratch.rt_type;
          rts.push(rt);
          rts_names.push(scratch.name);
        } else if (ch->cmp_symbol("body")) {
          src = ch->next;
        }
        cur = cur->next;
      }
      wnd.create_render_pass(name, width, height, deps_names.ptr, deps_names.size, rts_names.ptr,
                             rts.ptr, rts.size, src);
      return NULL;
    }
    if (prev != NULL) return prev->eval(l);
    return {NULL, false};
  }
};

IEvaluator *create_rendering_mode() {
  Renderign_Evaluator *eval = new Renderign_Evaluator();
  eval->init();
  return eval;
}

static int _init = [] {
  IEvaluator::add_mode(stref_s("rendering"), create_rendering_mode);
  return 0;
}();
