#define UTILS_TL_IMPL
#include "utils.hpp"
#define SCRIPT_IMPL
#include "script.hpp"

#include <functional>

#define SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS

#ifdef __linux__
#  define VK_USE_PLATFORM_XCB_KHR
#  include <SDL2/SDL.h>
#  include <SDL2/SDL_vulkan.h>
#  include <shaderc/shaderc.h>
#  include <spirv_cross/spirv_cross_c.h>
#  include <vulkan/vulkan.h>
#else
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <SDL.h>
#  include <SDL_vulkan.h>
#  include <shaderc/shaderc.h>
#  include <spirv_cross/spirv_cross_c.h>
#  include <vulkan/vulkan.h>
#endif

#include "rendering.hpp"

#include <spirv_cross/spirv_cross.hpp>

#define VK_ASSERT_OK(x)                                                                            \
  do {                                                                                             \
    VkResult __res = x;                                                                            \
    if (__res != VK_SUCCESS) {                                                                     \
      fprintf(stderr, "VkResult: %i\n", (i32)__res);                                               \
      TRAP;                                                                                        \
    }                                                                                              \
  } while (0)

namespace {
Pool<char> string_storage = Pool<char>::create(1 << 20);

string_ref relocate_cstr(string_ref old) {
  char *     new_ptr = string_storage.put(old.ptr, old.len + 1);
  string_ref new_ref = string_ref{new_ptr, old.len};
  new_ptr[old.len]   = '\0';
  return new_ref;
}

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

struct Mem_Chunk : public Ref_Cnt {

  VkDeviceMemory        mem              = VK_NULL_HANDLE;
  VkMemoryPropertyFlags prop_flags       = 0;
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
  void init(VkDevice device, u32 size, u32 heap_index, VkMemoryPropertyFlags prop_flags,
            u32 type_bits) {
    VkMemoryAllocateInfo info;
    MEMZERO(info);
    info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize  = size;
    info.memoryTypeIndex = heap_index;
    VK_ASSERT_OK(vkAllocateMemory(device, &info, nullptr, &mem));
    this->size             = size;
    this->prop_flags       = prop_flags;
    this->memory_type_bits = type_bits;
    this->cursor           = 0;
  }
  void release(VkDevice device) {
    vkFreeMemory(device, mem, NULL);
    memset(this, 0, sizeof(*this));
  }
  bool is_empty() const { return mem == VK_NULL_HANDLE; }
  bool has_space(u32 alignment, u32 req_size) {
    if (mem == VK_NULL_HANDLE) return false;
    if (ref_cnt == 0) cursor = 0;
    return req_size + ((cursor + alignment - 1) & ~(alignment - 1)) <= size;
  }
  u32 alloc(u32 alignment, u32 req_size) {
    if (ref_cnt == 0) cursor = 0;
    ASSERT_DEBUG((alignment & (alignment - 1)) == 0); // PoT
    // ASSERT_DEBUG(alignment < PAGE_SIZE || (alignment & (PAGE_SIZE - 1)) == 0); // PoT
    // if (alignment > PAGE_SIZE) {
    // u32 page_alignment = alignment / PAGE_SIZE;
    // ASSERT_DEBUG(((alignment - 1) & PAGE_SIZE) == 0); // 4kb bytes is
    // enough to align
    if (cursor != 0) { // Need to align
      cursor = ((cursor + alignment - 1) & ~(alignment - 1));
    }
    //}
    u32 offset = cursor;
    cursor += req_size;
    ASSERT_DEBUG(cursor <= size);
    ref_cnt++;
    return offset;
  }
};

struct BufferView_Flags {
  VkFormat     format;
  VkDeviceSize offset;
  VkDeviceSize range;
};

u64 hash_of(BufferView_Flags const &state) {
  return hash_of(string_ref{(char const *)&state, sizeof(state)});
}

struct Buffer : public Slot {
  string             name;
  ID                 mem_chunk_id;
  u32                mem_offset;
  VkBuffer           buffer;
  VkBufferCreateInfo create_info;
  VkAccessFlags      access_flags;
  InlineArray<ID, 8> views;

  void barrier(VkCommandBuffer cmd, VkAccessFlags new_access_flags) {
    VkBufferMemoryBarrier bar;
    MEMZERO(bar);
    bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bar.srcAccessMask       = access_flags;
    bar.dstAccessMask       = new_access_flags;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer              = buffer;
    bar.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 1, &bar, 0, NULL);
    access_flags = new_access_flags;
  }
  void init() {
    memset(this, 0, sizeof(*this));
    views.init();
  }
  void release() {
    views.release();
    name.release();
  }
};

static inline bool operator==(VkComponentMapping const &a, VkComponentMapping const &b) {
  return a.a == b.a && a.r == b.r && a.g == b.g && a.b == b.b;
}

static inline bool operator==(VkImageSubresourceRange const &a, VkImageSubresourceRange const &b) {
  return a.aspectMask == b.aspectMask && a.baseArrayLayer == b.baseArrayLayer &&
         a.baseMipLevel == b.baseMipLevel && a.layerCount == b.layerCount &&
         a.levelCount == b.levelCount;
}

struct ImageView_Flags {
  VkImageViewType         viewType;
  VkFormat                format;
  VkComponentMapping      components;
  VkImageSubresourceRange subresourceRange;
  bool                    operator==(ImageView_Flags const &that) {
    return viewType == that.viewType && format == that.format && components == that.components &&
           subresourceRange == that.subresourceRange;
  }
};

u64 hash_of(ImageView_Flags const &state) {
  return hash_of(string_ref{(char const *)&state, sizeof(state)});
}

struct Image_Info {
  VkImageType           imageType;
  VkFormat              format;
  VkExtent3D            extent;
  uint32_t              mipLevels;
  uint32_t              arrayLayers;
  VkSampleCountFlagBits samples;
  VkImageTiling         tiling;
  VkImageUsageFlags     usage;
  VkSharingMode         sharingMode;
};

VkAccessFlags to_vk_access_flags(u32 flags) {
  u32 out = 0;
  // clang-format off
  if (flags & rd::Access_Bits::UNIFORM_READ                     ) out |= VK_ACCESS_UNIFORM_READ_BIT                   ;
  if (flags & rd::Access_Bits::SHADER_READ                      ) out |= VK_ACCESS_SHADER_READ_BIT                    ;
  if (flags & rd::Access_Bits::SHADER_WRITE                     ) out |= VK_ACCESS_SHADER_WRITE_BIT                   ;
  if (flags & rd::Access_Bits::TRANSFER_READ                    ) out |= VK_ACCESS_TRANSFER_READ_BIT                  ;
  if (flags & rd::Access_Bits::TRANSFER_WRITE                   ) out |= VK_ACCESS_TRANSFER_WRITE_BIT                 ;
  if (flags & rd::Access_Bits::HOST_READ                        ) out |= VK_ACCESS_HOST_READ_BIT                      ;
  if (flags & rd::Access_Bits::HOST_WRITE                       ) out |= VK_ACCESS_HOST_WRITE_BIT                     ;
  if (flags & rd::Access_Bits::MEMORY_READ                      ) out |= VK_ACCESS_MEMORY_READ_BIT                    ;
  if (flags & rd::Access_Bits::MEMORY_WRITE                     ) out |= VK_ACCESS_MEMORY_WRITE_BIT                   ;
  // clang-format on
  return out;
}

VkImageLayout to_vk(rd::Image_Layout layout) {
  // clang-format off
  switch (layout) {
  case rd::Image_Layout::SHADER_READ_ONLY_OPTIMAL      : return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  case rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL     : return VK_IMAGE_LAYOUT_GENERAL;
  case rd::Image_Layout::TRANSFER_DST_OPTIMAL          : return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  case rd::Image_Layout::TRANSFER_SRC_OPTIMAL          : return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

VkFormat to_vk(rd::Format format) {
  // clang-format off
  switch (format) {
  case rd::Format::RGBA8_UNORM     : return VK_FORMAT_R8G8B8A8_UNORM      ;
  case rd::Format::RGBA8_SNORM     : return VK_FORMAT_R8G8B8A8_SNORM      ;
  case rd::Format::RGBA8_SRGBA     : return VK_FORMAT_R8G8B8A8_SRGB       ;
  case rd::Format::RGBA8_UINT      : return VK_FORMAT_R8G8B8A8_UINT       ;

  case rd::Format::RGB8_UNORM      : return VK_FORMAT_R8G8B8_UNORM        ;
  case rd::Format::RGB8_SNORM      : return VK_FORMAT_R8G8B8_SNORM        ;
  case rd::Format::RGB8_SRGBA      : return VK_FORMAT_R8G8B8_SRGB         ;
  case rd::Format::RGB8_UINT       : return VK_FORMAT_R8G8B8_UINT         ;

  case rd::Format::RGBA32_FLOAT    : return VK_FORMAT_R32G32B32A32_SFLOAT ;
  case rd::Format::RGB32_FLOAT     : return VK_FORMAT_R32G32B32_SFLOAT    ;
  case rd::Format::RG32_FLOAT      : return VK_FORMAT_R32G32_SFLOAT       ;
  case rd::Format::R32_FLOAT       : return VK_FORMAT_R32_SFLOAT          ;
  case rd::Format::R16_FLOAT       : return VK_FORMAT_R16_SFLOAT          ;
  case rd::Format::D32_FLOAT       : return VK_FORMAT_D32_SFLOAT          ;
  case rd::Format::R32_UINT        : return VK_FORMAT_R32_UINT            ;
  case rd::Format::R16_UINT        : return VK_FORMAT_R16_UINT            ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

rd::Format from_vk(VkFormat format) {
  // clang-format off
  switch (format) {
  case VK_FORMAT_B8G8R8A8_UNORM      : return rd::Format::BGRA8_UNORM     ;
  case VK_FORMAT_B8G8R8_UNORM        : return rd::Format::BGR8_UNORM      ;

  case VK_FORMAT_R8G8B8A8_UNORM      : return rd::Format::RGBA8_UNORM     ;
  case VK_FORMAT_R8G8B8A8_SNORM      : return rd::Format::RGBA8_SNORM     ;
  case VK_FORMAT_R8G8B8A8_SRGB       : return rd::Format::RGBA8_SRGBA     ;
  case VK_FORMAT_R8G8B8A8_UINT       : return rd::Format::RGBA8_UINT      ;

  case VK_FORMAT_R8G8B8_UNORM        : return rd::Format::RGB8_UNORM      ;
  case VK_FORMAT_R8G8B8_SNORM        : return rd::Format::RGB8_SNORM      ;
  case VK_FORMAT_R8G8B8_SRGB         : return rd::Format::RGB8_SRGBA      ;
  case VK_FORMAT_R8G8B8_UINT         : return rd::Format::RGB8_UINT       ;

  case VK_FORMAT_R32G32B32A32_SFLOAT : return rd::Format::RGBA32_FLOAT    ;
  case VK_FORMAT_R32G32B32_SFLOAT    : return rd::Format::RGB32_FLOAT     ;
  case VK_FORMAT_R32G32_SFLOAT       : return rd::Format::RG32_FLOAT      ;
  case VK_FORMAT_R32_SFLOAT          : return rd::Format::R32_FLOAT       ;
  case VK_FORMAT_D32_SFLOAT          : return rd::Format::D32_FLOAT       ;
  case VK_FORMAT_R32_UINT            : return rd::Format::R32_UINT        ;
  case VK_FORMAT_R16_UINT            : return rd::Format::R16_UINT        ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

u32 get_format_size(VkFormat format) {
  // clang-format off
  switch (format) {
  case VK_FORMAT_B8G8R8A8_UNORM      : return 4     ;
  case VK_FORMAT_B8G8R8_UNORM        : return 4     ;
                                              
  case VK_FORMAT_R8G8B8A8_UNORM      : return 4     ;
  case VK_FORMAT_R8G8B8A8_SNORM      : return 4     ;
  case VK_FORMAT_R8G8B8A8_SRGB       : return 4     ;
  case VK_FORMAT_R8G8B8A8_UINT       : return 4     ;
                                              
  case VK_FORMAT_R8G8B8_UNORM        : return 4     ;
  case VK_FORMAT_R8G8B8_SNORM        : return 4     ;
  case VK_FORMAT_R8G8B8_SRGB         : return 4     ;
  case VK_FORMAT_R8G8B8_UINT         : return 4     ;
                                              
  case VK_FORMAT_R32G32B32A32_SFLOAT : return 16    ;
  case VK_FORMAT_R32G32B32_SFLOAT    : return 12    ;
  case VK_FORMAT_R32G32_SFLOAT       : return 8     ;
  case VK_FORMAT_R32_SFLOAT          : return 4     ;
  case VK_FORMAT_D32_SFLOAT          : return 4     ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

struct Image : public Slot {
  ID                    mem_chunk_id;
  u32                   mem_offset;
  VkImageLayout         layout;
  VkAccessFlags         access_flags;
  VkImageAspectFlags    aspect;
  VkImage               image;
  Image_Info            info;
  InlineArray<ID, 0x10> views;
  u32                   getbpp() const { return get_format_size(info.format); }
  void                  init() {
    memset(this, 0, sizeof(*this));
    views.init();
  }
  void release() { views.release(); }
  bool is_depth_image() {
    switch (info.format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return true;
    default: return false;
    }
  }
  void barrier(VkCommandBuffer cmd, VkAccessFlags new_access_flags, VkImageLayout new_layout) {
    // if (new_access_flags == access_flags && new_layout == layout) return;
    VkImageMemoryBarrier bar;
    MEMZERO(bar);
    bar.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.srcAccessMask                   = access_flags;
    bar.dstAccessMask                   = new_access_flags;
    bar.oldLayout                       = layout;
    bar.newLayout                       = new_layout;
    bar.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar.image                           = image;
    bar.subresourceRange.aspectMask     = aspect;
    bar.subresourceRange.baseArrayLayer = 0;
    bar.subresourceRange.baseMipLevel   = 0;
    bar.subresourceRange.layerCount     = info.arrayLayers;
    bar.subresourceRange.levelCount     = info.mipLevels;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
    layout       = new_layout;
    access_flags = new_access_flags;
  }
};

struct ImageView : public Slot {
  ID              img_id;
  VkImageView     view;
  ImageView_Flags flags;
};

struct BufferView : public Slot {
  ID               buf_id;
  VkBufferView     view;
  BufferView_Flags flags;
};

struct Sampler : public Slot {
  rd::Sampler_Create_Info create_info;
  VkSampler               sampler;
};

struct Shader_Descriptor {
  string             name;
  u32                set;
  u32                binding;
  VkDescriptorType   descriptorType;
  u32                descriptorCount;
  VkShaderStageFlags stageFlags;

  void init(string             name,            //
            u32                set,             //
            uint32_t           binding,         //
            VkDescriptorType   descriptorType,  //
            uint32_t           descriptorCount, //
            VkShaderStageFlags stageFlags) {
    this->name            = name;
    this->set             = set;
    this->binding         = binding;
    this->descriptorType  = descriptorType;
    this->descriptorCount = descriptorCount;
    this->stageFlags      = stageFlags;
  }
  void release() {
    name.release();
    memset(this, 0, sizeof(*this));
  }
};

struct Shader_Descriptor_Set {
  Array<Shader_Descriptor> descriptors;
  void                     init() { descriptors.init(); }
  void                     release() {
    ito(descriptors.size) descriptors[i].release();
    descriptors.release();
  }
};

static Array<u32> compile_glsl(VkDevice device, string_ref text, shaderc_shader_kind kind,
                               Pair<string_ref, string_ref> *defines, size_t num_defines) {
  shaderc_compiler_t        compiler = shaderc_compiler_initialize();
  shaderc_compile_options_t options  = shaderc_compile_options_initialize();

  ito(num_defines) shaderc_compile_options_add_macro_definition(
      options, defines[i].first.ptr, defines[i].first.len, defines[i].second.ptr,
      defines[i].second.len);

  shaderc_compile_options_set_source_language(options, shaderc_source_language_glsl);
#ifndef NDEBUG
  shaderc_compile_options_set_generate_debug_info(options);
#endif
  // shaderc_compile_options_set_optimization_level(options,
  // shaderc_optimization_level_performance);
  shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_3);
  shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_2);
  shaderc_compilation_result_t result =
      shaderc_compile_into_spv(compiler, text.ptr, text.len, kind, "tmp", "main", options);
  // fprintf(stderr, "%.*s\n", STRF(text));
  // fflush(stderr);
  defer({
    shaderc_result_release(result);
    shaderc_compiler_release(compiler);
    shaderc_compile_options_release(options);
  });
  if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
    u32 len = 1;
    fprintf(stderr, "%i:", len);
    ito(text.len) {
      fprintf(stderr, "%c", text.ptr[i]);
      if (text.ptr[i] == '\n') {
        len += 1;
        fprintf(stderr, "%i:", len);
      }
    }

    fprintf(stderr, shaderc_result_get_error_message(result));
    TRAP;
  }
  size_t     len      = shaderc_result_get_length(result);
  u32 *      bytecode = (u32 *)shaderc_result_get_bytes(result);
  Array<u32> out;
  out.init(bytecode, len / 4);
  return out;
}

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
      } else if (format == stref_s("R16_FLOAT")) {
        return stref_s("r16f");
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
      } else if (format == stref_s("R16_FLOAT")) {
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
#define saturate(x) (x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x))

bvec3 greater(vec3 a, vec3 b) {
  return bvec3(a.x > b.x, a.y > b.y, a.z > b.z);
}
bool any(bvec3 a) {
  return a.x || a.y || a.z;
}

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

struct Shader_Info : public Slot {
  rd::Stage_t stage;
  u64         hash;
  Array<u32>  bytecode;
  void        init(rd::Stage_t stage, u64 hash, Array<u32> bytecode) {
    this->hash     = hash;
    this->stage    = stage;
    this->bytecode = bytecode;
  }
  VkShaderStageFlags get_stage_bits() const {
    switch (stage) {
    case rd::Stage_t::COMPUTE: return VK_SHADER_STAGE_COMPUTE_BIT;
    case rd::Stage_t::PIXEL: return VK_SHADER_STAGE_FRAGMENT_BIT;
    case rd::Stage_t::VERTEX: return VK_SHADER_STAGE_VERTEX_BIT;
    default: UNIMPLEMENTED; ;
    }
  }
  void           release() { bytecode.release(); }
  VkShaderModule compile(VkDevice device) {
    VkShaderModuleCreateInfo info;
    MEMZERO(info);
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = bytecode.size * 4;
    info.flags    = 0;
    info.pCode    = bytecode.ptr;
    VkShaderModule module;
    VK_ASSERT_OK(vkCreateShaderModule(device, &info, NULL, &module));
    return module;
  }
};

struct Render_Pass : public Slot {
  string                       name;
  VkRenderPass                 pass;
  VkFramebuffer                fb;
  u32                          width, height;
  InlineArray<ID, 8>           rts_views;
  ID                           depth_target_view;
  InlineArray<VkClearValue, 9> clear_values;
  rd::Render_Pass_Create_Info  create_info;
  void                         init() {
    memset(this, 0, sizeof(*this));
    rts_views.init();
    clear_values.init();
    create_info.reset();
  }
  void release(VkDevice device) {
    rts_views.release();
    create_info.reset();
    clear_values.release();
    name.release();
    vkDestroyRenderPass(device, pass, NULL);
    vkDestroyFramebuffer(device, fb, NULL);
    memset(this, 0, sizeof(*this));
  }
};

VkCompareOp to_vk(rd::Cmp cmp) {
  switch (cmp) {
  case rd::Cmp::EQ: return VK_COMPARE_OP_EQUAL;
  case rd::Cmp::GE: return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case rd::Cmp::GT: return VK_COMPARE_OP_GREATER;
  case rd::Cmp::LE: return VK_COMPARE_OP_LESS_OR_EQUAL;
  case rd::Cmp::LT: return VK_COMPARE_OP_LESS;
  default: UNIMPLEMENTED;
  }
}

VkFilter to_vk(rd::Filter cmp) {
  switch (cmp) {
  case rd::Filter::NEAREST: return VK_FILTER_NEAREST;
  case rd::Filter::LINEAR: return VK_FILTER_LINEAR;
  default: UNIMPLEMENTED;
  }
}

VkSamplerAddressMode to_vk(rd::Address_Mode cmp) {
  switch (cmp) {
  case rd::Address_Mode::REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case rd::Address_Mode::CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case rd::Address_Mode::MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  default: UNIMPLEMENTED;
  }
}

VkCullModeFlags to_vk(rd::Cull_Mode cmp) {
  switch (cmp) {
  case rd::Cull_Mode::BACK: return VK_CULL_MODE_BACK_BIT;
  case rd::Cull_Mode::FRONT: return VK_CULL_MODE_FRONT_BIT;
  case rd::Cull_Mode::NONE: return VK_CULL_MODE_NONE;
  default: UNIMPLEMENTED;
  }
}

VkFrontFace to_vk(rd::Front_Face cmp) {
  switch (cmp) {
  case rd::Front_Face::CCW: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
  case rd::Front_Face::CW: return VK_FRONT_FACE_CLOCKWISE;
  default: UNIMPLEMENTED;
  }
}

VkPolygonMode to_vk(rd::Polygon_Mode cmp) {
  switch (cmp) {
  case rd::Polygon_Mode::FILL: return VK_POLYGON_MODE_FILL;
  case rd::Polygon_Mode::LINE: return VK_POLYGON_MODE_LINE;
  default: UNIMPLEMENTED;
  }
}

VkSampleCountFlagBits to_sample_bit(u32 num_samples) {
  switch (num_samples) {
  case 1: return VK_SAMPLE_COUNT_1_BIT;
  case 2: return VK_SAMPLE_COUNT_2_BIT;
  case 4: return VK_SAMPLE_COUNT_4_BIT;
  case 8: return VK_SAMPLE_COUNT_8_BIT;
  case 16: return VK_SAMPLE_COUNT_16_BIT;
  case 32: return VK_SAMPLE_COUNT_32_BIT;
  default: UNIMPLEMENTED;
  }
}

VkClearValue to_vk(rd::Clear_Color cl) {
  VkClearValue out;
  MEMZERO(out);
  out.color.float32[0] = cl.r;
  out.color.float32[1] = cl.g;
  out.color.float32[2] = cl.b;
  out.color.float32[3] = cl.a;
  return out;
}

VkClearValue to_vk(rd::Clear_Depth cl) {
  VkClearValue out;
  MEMZERO(out);
  out.depthStencil.depth   = cl.d;
  out.depthStencil.stencil = 0;
  return out;
}

VkBlendFactor to_vk(rd::Blend_Factor bf) {
  // clang-format off
  switch (bf) {
  case rd::Blend_Factor::ZERO                     : return VK_BLEND_FACTOR_ZERO                     ;
  case rd::Blend_Factor::ONE                      : return VK_BLEND_FACTOR_ONE                      ;
  case rd::Blend_Factor::SRC_COLOR                : return VK_BLEND_FACTOR_SRC_COLOR                ;
  case rd::Blend_Factor::ONE_MINUS_SRC_COLOR      : return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR      ;
  case rd::Blend_Factor::DST_COLOR                : return VK_BLEND_FACTOR_DST_COLOR                ;
  case rd::Blend_Factor::ONE_MINUS_DST_COLOR      : return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR      ;
  case rd::Blend_Factor::SRC_ALPHA                : return VK_BLEND_FACTOR_SRC_ALPHA                ;
  case rd::Blend_Factor::ONE_MINUS_SRC_ALPHA      : return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA      ;
  case rd::Blend_Factor::DST_ALPHA                : return VK_BLEND_FACTOR_DST_ALPHA                ;
  case rd::Blend_Factor::ONE_MINUS_DST_ALPHA      : return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA      ;
  case rd::Blend_Factor::CONSTANT_COLOR           : return VK_BLEND_FACTOR_CONSTANT_COLOR           ;
  case rd::Blend_Factor::ONE_MINUS_CONSTANT_COLOR : return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR ;
  case rd::Blend_Factor::CONSTANT_ALPHA           : return VK_BLEND_FACTOR_CONSTANT_ALPHA           ;
  case rd::Blend_Factor::ONE_MINUS_CONSTANT_ALPHA : return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

VkBlendOp to_vk(rd::Blend_OP bf) {
  // clang-format off
  switch (bf) {
  case rd::Blend_OP::ADD                     : return VK_BLEND_OP_ADD;
  case rd::Blend_OP::MAX                     : return VK_BLEND_OP_MAX;
  case rd::Blend_OP::MIN                     : return VK_BLEND_OP_MIN;
  case rd::Blend_OP::REVERSE_SUBTRACT        : return VK_BLEND_OP_REVERSE_SUBTRACT;
  case rd::Blend_OP::SUBTRACT                : return VK_BLEND_OP_SUBTRACT;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

u32 to_vk_buffer_usage_bits(u32 usage_bits) {
  u32 usage = 0;
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST) {
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_INDIRECT_ARGUMENTS) {
    usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER) {
    usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER) {
    usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_UAV) {
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (usage_bits & (i32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER) {
    usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  return usage;
}

u32 to_vk_image_usage_bits(u32 usage_flags) {
  u32 usage = 0;
  if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_RT) {
    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_DT) {
    usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  }
  if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_SAMPLED) {
    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_UAV) {
    usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST) {
    usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  }
  if (usage_flags & (i32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC) {
    usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  return usage;
}

u32 to_vk_memory_bits(u32 mem_bits) {
  u32 prop_flags = 0;
  if (mem_bits & (i32)rd::Memory_Bits::HOST_VISIBLE) {
    prop_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  if (mem_bits & (i32)rd::Memory_Bits::COHERENT) {
    prop_flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (mem_bits & (i32)rd::Memory_Bits::DEVICE_LOCAL) {
    prop_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }
  if (mem_bits & (i32)rd::Memory_Bits::HOST_CACHED) {
    prop_flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  return prop_flags;
}

VkBool32 to_vk(bool b) { return b ? VK_TRUE : VK_FALSE; }

VkPrimitiveTopology to_vk(rd::Primitive p) {
  switch (p) {
  case rd::Primitive::TRIANGLE_LIST: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case rd::Primitive::TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  case rd::Primitive::LINE_LIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  default: UNIMPLEMENTED;
  }
}

struct Graphics_Pipeline_State {
  VkVertexInputBindingDescription     bindings[0x10];
  VkVertexInputAttributeDescription   attributes[0x10];
  VkPrimitiveTopology                 topology;
  rd::RS_State                        rs_state;
  rd::DS_State                        ds_state;
  ID                                  ps, vs;
  u32                                 num_rts;
  VkPipelineColorBlendAttachmentState blend_states[8];
  rd::MS_State                        ms_state;

  VkPipelineDepthStencilStateCreateInfo get_ds_create_info() {
    VkPipelineDepthStencilStateCreateInfo ds_create_info;
    MEMZERO(ds_create_info);
    ds_create_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds_create_info.depthTestEnable   = to_vk(ds_state.enable_depth_test);
    ds_create_info.depthCompareOp    = to_vk(ds_state.cmp_op);
    ds_create_info.depthWriteEnable  = to_vk(ds_state.enable_depth_write);
    ds_create_info.maxDepthBounds    = ds_state.max_depth;
    ds_create_info.stencilTestEnable = VK_FALSE;
    return ds_create_info;
  }
  VkPipelineRasterizationStateCreateInfo get_rs_create_info() {
    VkPipelineRasterizationStateCreateInfo rs_create_info;
    MEMZERO(rs_create_info);
    rs_create_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs_create_info.cullMode        = to_vk(rs_state.cull_mode);
    rs_create_info.frontFace       = to_vk(rs_state.front_face);
    rs_create_info.lineWidth       = 1.0f;
    rs_create_info.polygonMode     = to_vk(rs_state.polygon_mode);
    rs_create_info.depthBiasEnable = true;
    return rs_create_info;
  }
  VkPipelineMultisampleStateCreateInfo get_ms_create_info() {
    VkPipelineMultisampleStateCreateInfo ms_create_info;
    MEMZERO(ms_create_info);
    ms_create_info.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms_create_info.rasterizationSamples  = to_sample_bit(ms_state.num_samples);
    ms_create_info.alphaToCoverageEnable = to_vk(ms_state.alpha_to_coverage);
    ms_create_info.alphaToOneEnable      = to_vk(ms_state.alpha_to_one);
    ms_create_info.minSampleShading      = ms_state.min_sample_shading;
    ms_create_info.pSampleMask           = &ms_state.sample_mask;
    ms_create_info.sampleShadingEnable   = to_vk(ms_state.sample_shading);
    return ms_create_info;
  }
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

struct Descriptor_Pool {
  VkDevice         device;
  VkDescriptorPool pool;
  void             init(VkDevice device) {
    VkDescriptorPoolSize aPoolSizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 100000},                //
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100000}, //
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100000},          //
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100000},         //
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100000},         //
    };
    VkDescriptorPoolCreateInfo info;
    MEMZERO(info);
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = ARRAY_SIZE(aPoolSizes);
    info.pPoolSizes    = aPoolSizes;
    info.maxSets       = 1 << 14;
    VK_ASSERT_OK(vkCreateDescriptorPool(device, &info, NULL, &pool));
    this->device = device;
  }
  VkDescriptorSet allocate(VkDescriptorSetLayout layout) {
    VkDescriptorSet             set;
    VkDescriptorSetAllocateInfo info;
    MEMZERO(info);
    info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool     = pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts        = &layout;
    info.pNext              = NULL;
    VK_ASSERT_OK(vkAllocateDescriptorSets(device, &info, &set));
    return set;
  }
  void reset() { vkResetDescriptorPool(device, pool, 0); }
  void release() { vkDestroyDescriptorPool(device, pool, NULL); }
};

struct Shader_Reflection {
  template <typename K, typename V> using Table = Hash_Table<K, V, Default_Allocator, 0x10>;
  Table<u32, Table<u32, Shader_Descriptor>> set_table;
  u32                                       push_constants_size;
  void                                      init() {
    set_table.init();
    push_constants_size = 0;
  }
  void release(VkDevice device) {
    set_table.iter_values([](Table<u32, Shader_Descriptor> &val) { val.release(); });
    set_table.release();
  }

  template <typename T> void merge_into(T &table) {
    set_table.iter_pairs([&](u32 set_index, Table<u32, Shader_Descriptor> &binding_table) {
      if (table.contains(set_index)) {
        Table<u32, Shader_Descriptor> &merg = table.get(set_index);
        binding_table.iter_pairs([&](u32 binding_index, Shader_Descriptor &desc) {
          if (!merg.contains(binding_index)) {
            merg.insert(binding_index, desc);
          }
        });
      } else {
        table.insert(set_index, binding_table.clone());
      }
    });
  }

  template <typename T>
  static void create_layouts(VkDevice device, Table<u32, Table<u32, Shader_Descriptor>> &set_table,
                             T &out) {
    set_table.iter_pairs([&](u32 set_index, Table<u32, Shader_Descriptor> &binding_table) {
      constexpr u32                MAX_BINDINGS = 0x40;
      VkDescriptorBindingFlags     binding_flags[MAX_BINDINGS];
      u32                          num_bindings = 0;
      VkDescriptorSetLayoutBinding set_bindings[MAX_BINDINGS];
      binding_table.iter_values([&](Shader_Descriptor &val) {
        VkDescriptorSetLayoutBinding binding_info;
        MEMZERO(binding_info);
        binding_info.binding            = val.binding;
        binding_info.descriptorCount    = val.descriptorCount;
        binding_info.descriptorType     = val.descriptorType;
        binding_info.pImmutableSamplers = NULL;
        binding_info.stageFlags         = VK_SHADER_STAGE_ALL;
        set_bindings[num_bindings++]    = binding_info;
      });
      VkDescriptorSetLayoutBindingFlagsCreateInfo binding_infos;

      ito(num_bindings) {
        if (set_bindings[i].descriptorCount > 1) {
          binding_flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        } else {
          binding_flags[i] = 0;
        }
      }
      ASSERT_DEBUG(num_bindings < MAX_BINDINGS);
      binding_infos.bindingCount  = num_bindings;
      binding_infos.pBindingFlags = &binding_flags[0];
      binding_infos.pNext         = NULL;
      binding_infos.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

      VkDescriptorSetLayoutCreateInfo set_layout_create_info;
      MEMZERO(set_layout_create_info);
      set_layout_create_info.bindingCount = num_bindings;
      set_layout_create_info.pBindings    = &set_bindings[0];
      set_layout_create_info.flags        = 0
          //  | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
          ;
      set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      set_layout_create_info.pNext = (void *)&binding_infos;
      VkDescriptorSetLayout set_layout;
      VK_ASSERT_OK(vkCreateDescriptorSetLayout(device, &set_layout_create_info, NULL, &set_layout));
      if (set_index >= out.size) {
        out.resize(set_index + 1);
      }
      out[set_index] = set_layout;
    });
    // Put empty set layouts
    ito(out.size) {
      if (out[i] == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutCreateInfo set_layout_create_info;
        MEMZERO(set_layout_create_info);
        set_layout_create_info.bindingCount = 0;
        set_layout_create_info.pBindings    = NULL;
        set_layout_create_info.flags        = 0;
        set_layout_create_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        VkDescriptorSetLayout set_layout;
        VK_ASSERT_OK(
            vkCreateDescriptorSetLayout(device, &set_layout_create_info, NULL, &set_layout));
        out[i] = set_layout;
      }
    }
  }
};

Shader_Reflection reflect_shader(Shader_Info const &info) {
  spirv_cross::Compiler        comp({info.bytecode.ptr, info.bytecode.ptr + info.bytecode.size});
  spirv_cross::ShaderResources res = comp.get_shader_resources();
  Shader_Reflection            out;
  out.init();
  auto handle_resource = [&](VkDescriptorType desc_type, spirv_cross::Resource &item) {
    Shader_Descriptor     desc;
    spirv_cross::SPIRType type_obj      = comp.get_type(item.type_id);
    spirv_cross::SPIRType base_type_obj = comp.get_type(item.base_type_id);
    auto set             = comp.get_decoration(item.id, spv::Decoration::DecorationDescriptorSet);
    desc.name            = make_string(item.name.c_str());
    desc.set             = set;
    desc.binding         = comp.get_decoration(item.id, spv::Decoration::DecorationBinding);
    desc.descriptorCount = 1;
    if (type_obj.array.size() != 0) {
      ASSERT_ALWAYS(type_obj.array.size() == 1);
      desc.descriptorCount = type_obj.array[0];
    }
    desc.stageFlags     = info.get_stage_bits();
    desc.descriptorType = desc_type;
    if (out.set_table.contains(set) == false) {
      Shader_Reflection::Table<u32, Shader_Descriptor> descset;
      descset.init();
      out.set_table.insert(set, descset);
    }
    Shader_Reflection::Table<u32, Shader_Descriptor> &descset = out.set_table.get_ref(set);
    ASSERT_DEBUG(descset.contains(desc.binding) == false);
    descset.insert(desc.binding, desc);
  };
  for (auto &item : res.storage_buffers) {
    handle_resource(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, item);
  }
  for (auto &item : res.sampled_images) {
    handle_resource(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, item);
  }
  for (auto &item : res.separate_samplers) {
    handle_resource(VK_DESCRIPTOR_TYPE_SAMPLER, item);
  }
  for (auto &item : res.separate_images) {
    handle_resource(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, item);
  }
  for (auto &item : res.storage_images) {
    handle_resource(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, item);
  }
  for (auto &item : res.uniform_buffers) {
    handle_resource(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, item);
  }
  for (auto &item : res.push_constant_buffers) {
    ASSERT_DEBUG(res.push_constant_buffers.size() == 1);
    spirv_cross::SPIRType type_obj      = comp.get_type(item.type_id);
    spirv_cross::SPIRType base_type_obj = comp.get_type(item.base_type_id);
    out.push_constants_size             = comp.get_declared_struct_size(type_obj);
  }
  return out;
}

struct Graphics_Pipeline_Wrapper : public Slot {
  VkPipelineLayout                      pipeline_layout;
  VkPipeline                            pipeline;
  VkShaderModule                        ps_module;
  VkShaderModule                        vs_module;
  InlineArray<VkDescriptorSetLayout, 8> set_layouts;
  Hash_Set<Pair<u32, u32>>              bindings;
  u32                                   push_constants_size;

  void release(VkDevice device) {
    ito(set_layouts.size) {
      if (set_layouts[i] != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
    }
    set_layouts.release();
    bindings.release();
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, vs_module, NULL);
    vkDestroyShaderModule(device, ps_module, NULL);
    MEMZERO(*this);
  }

  void init(VkDevice                 device,    //
            Render_Pass &            pass,      //
            Shader_Info &            vs_shader, //
            Shader_Info &            ps_shader, //
            Graphics_Pipeline_State &pipeline_info) {
    MEMZERO(*this);
    (void)pipeline_info;
    bindings.init();
    MEMZERO(set_layouts);
    push_constants_size = 0;
    VkPipelineShaderStageCreateInfo stages[2];
    Shader_Reflection::Table<u32, Shader_Reflection::Table<u32, Shader_Descriptor>>
        merged_set_table;
    merged_set_table.init();
    defer(merged_set_table.release());
    {
      vs_module = vs_shader.compile(device);
      VkPipelineShaderStageCreateInfo stage;
      MEMZERO(stage);
      stage.sType                     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage                     = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module                    = vs_module;
      stage.pName                     = "main";
      stages[0]                       = stage;
      Shader_Reflection vs_reflection = reflect_shader(vs_shader);
      push_constants_size             = MAX(vs_reflection.push_constants_size, push_constants_size);
      vs_reflection.merge_into(merged_set_table);
      defer(vs_reflection.release(device));
    }
    {
      ps_module = ps_shader.compile(device);
      VkPipelineShaderStageCreateInfo stage;
      MEMZERO(stage);
      stage.sType                     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage                     = VK_SHADER_STAGE_FRAGMENT_BIT;
      stage.module                    = ps_module;
      stage.pName                     = "main";
      stages[1]                       = stage;
      Shader_Reflection ps_reflection = reflect_shader(ps_shader);
      push_constants_size             = MAX(ps_reflection.push_constants_size, push_constants_size);
      ps_reflection.merge_into(merged_set_table);
      defer(ps_reflection.release(device));
    }
    merged_set_table.iter_pairs(
        [&](u32 set, Shader_Reflection::Table<u32, Shader_Descriptor> &bindings) {
          bindings.iter_pairs([&](u32 index, Shader_Descriptor &binding) {
            this->bindings.insert({set, index});
          });
        });
    Shader_Reflection::create_layouts(device, merged_set_table, set_layouts);
    {
      VkPipelineLayoutCreateInfo pipe_layout_info;
      MEMZERO(pipe_layout_info);
      pipe_layout_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipe_layout_info.pSetLayouts    = &set_layouts[0];
      pipe_layout_info.setLayoutCount = set_layouts.size;
      VkPushConstantRange push_range;
      push_range.offset     = 0;
      push_range.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
      push_range.size       = push_constants_size;
      ;
      if (push_range.size > 0) {
        pipe_layout_info.pPushConstantRanges    = &push_range;
        pipe_layout_info.pushConstantRangeCount = 1;
      }
      VK_ASSERT_OK(vkCreatePipelineLayout(device, &pipe_layout_info, NULL, &pipeline_layout));
    }
    {
      VkGraphicsPipelineCreateInfo info;
      MEMZERO(info);
      info.sType  = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      info.layout = pipeline_layout;

      VkPipelineColorBlendStateCreateInfo blend_create_info;
      MEMZERO(blend_create_info);
      blend_create_info.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
      blend_create_info.attachmentCount = pipeline_info.num_rts;
      blend_create_info.logicOpEnable   = VK_FALSE;
      blend_create_info.pAttachments    = &pipeline_info.blend_states[0];
      info.pColorBlendState             = &blend_create_info;

      VkPipelineDepthStencilStateCreateInfo ds_create_info = pipeline_info.get_ds_create_info();
      info.pDepthStencilState                              = &ds_create_info;

      VkViewport viewports[] = {VkViewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}};
      VkRect2D   scissors[]  = {VkRect2D{{0, 0}, {1, 1}}};
      VkPipelineViewportStateCreateInfo vp_create_info;
      MEMZERO(vp_create_info);
      vp_create_info.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
      vp_create_info.pViewports    = viewports;
      vp_create_info.viewportCount = 1;
      vp_create_info.pScissors     = scissors;
      vp_create_info.scissorCount  = 1;
      info.pViewportState          = &vp_create_info;

      VkPipelineRasterizationStateCreateInfo rs_create_info = pipeline_info.get_rs_create_info();
      info.pRasterizationState                              = &rs_create_info;

      VkDynamicState dynamic_states[] = {
          VK_DYNAMIC_STATE_VIEWPORT,
          VK_DYNAMIC_STATE_SCISSOR,
          VK_DYNAMIC_STATE_DEPTH_BIAS,
          VK_DYNAMIC_STATE_LINE_WIDTH,
      };
      VkPipelineDynamicStateCreateInfo dy_create_info;
      MEMZERO(dy_create_info);
      dy_create_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dy_create_info.dynamicStateCount = ARRAY_SIZE(dynamic_states);
      dy_create_info.pDynamicStates    = dynamic_states;
      info.pDynamicState               = &dy_create_info;

      VkPipelineMultisampleStateCreateInfo ms_state = pipeline_info.get_ms_create_info();
      info.pMultisampleState                        = &ms_state;

      VkPipelineInputAssemblyStateCreateInfo ia_create_info;
      MEMZERO(ia_create_info);
      ia_create_info.sType     = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      ia_create_info.topology  = pipeline_info.topology;
      info.pInputAssemblyState = &ia_create_info;

      VkPipelineVertexInputStateCreateInfo vs_create_info;
      MEMZERO(vs_create_info);
      VkVertexInputBindingDescription   bindings[0x10];
      VkVertexInputAttributeDescription attributes[0x10];
      u32                               num_attributes = 0;
      u32                               num_bindings   = 0;
      ito(0x10) {
        if (pipeline_info.attributes[i].format != VK_FORMAT_UNDEFINED) {
          attributes[num_attributes++] = pipeline_info.attributes[i];
        }
      }
      ito(0x10) {
        if (pipeline_info.bindings[i].stride != 0) {
          bindings[num_bindings++] = pipeline_info.bindings[i];
        }
      }
      vs_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vs_create_info.pVertexAttributeDescriptions    = attributes;
      vs_create_info.vertexAttributeDescriptionCount = num_attributes;
      vs_create_info.pVertexBindingDescriptions      = bindings;
      vs_create_info.vertexBindingDescriptionCount   = num_bindings;
      info.pVertexInputState                         = &vs_create_info;

      info.renderPass = pass.pass;
      info.pStages    = stages;
      info.stageCount = 2;
      VK_ASSERT_OK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
    }
  }
};

struct Fence : public Slot {
  VkFence fence;
  void    init() { MEMZERO(*this); }
  void    release() {}
};

struct Event : public Slot {
  VkEvent event;
  void    init() { MEMZERO(*this); }
  void    release() {}
};

struct Semaphore : public Slot {
  VkSemaphore sem;
  void        init() { MEMZERO(*this); }
  void        release() {}
};

struct CommandBuffer : public Slot {
  VkCommandPool   pool;
  VkCommandBuffer cmd;
  void            init() { MEMZERO(*this); }
  void            release() {}
};

struct Compute_Pipeline_Wrapper : public Slot {
  VkPipelineLayout                      pipeline_layout;
  VkPipeline                            pipeline;
  VkShaderModule                        cs_module;
  InlineArray<VkDescriptorSetLayout, 8> set_layouts;
  Hash_Set<Pair<u32, u32>>              bindings;
  u32                                   push_constants_size;

  void release(VkDevice device) {
    ito(set_layouts.size) {
      if (set_layouts[i] != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
    }
    set_layouts.release();
    bindings.release();
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, cs_module, NULL);
    MEMZERO(*this);
  }

  void init(VkDevice     device, //
            Shader_Info &cs_shader) {
    MEMZERO(*this);
    MEMZERO(set_layouts);
    bindings.init();
    push_constants_size = 0;
    VkPipelineShaderStageCreateInfo stages[1];
    Shader_Reflection::Table<u32, Shader_Reflection::Table<u32, Shader_Descriptor>>
        merged_set_table;
    merged_set_table.init();
    defer(merged_set_table.release());
    {
      cs_module = cs_shader.compile(device);
      VkPipelineShaderStageCreateInfo stage;
      MEMZERO(stage);
      stage.sType                     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage                     = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module                    = cs_module;
      stage.pName                     = "main";
      stages[0]                       = stage;
      Shader_Reflection cs_reflection = reflect_shader(cs_shader);
      push_constants_size             = MAX(cs_reflection.push_constants_size, push_constants_size);
      cs_reflection.merge_into(merged_set_table);
      defer(cs_reflection.release(device));
    }
    merged_set_table.iter_pairs(
        [&](u32 set, Shader_Reflection::Table<u32, Shader_Descriptor> &bindings) {
          bindings.iter_pairs([&](u32 index, Shader_Descriptor &binding) {
            this->bindings.insert({set, index});
          });
        });
    Shader_Reflection::create_layouts(device, merged_set_table, set_layouts);
    {
      VkPipelineLayoutCreateInfo pipe_layout_info;
      MEMZERO(pipe_layout_info);
      pipe_layout_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipe_layout_info.pSetLayouts    = &set_layouts[0];
      pipe_layout_info.setLayoutCount = set_layouts.size;
      VkPushConstantRange push_range;
      push_range.offset     = 0;
      push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      push_range.size       = push_constants_size;
      ;
      if (push_range.size > 0) {
        pipe_layout_info.pPushConstantRanges    = &push_range;
        pipe_layout_info.pushConstantRangeCount = 1;
      }
      VK_ASSERT_OK(vkCreatePipelineLayout(device, &pipe_layout_info, NULL, &pipeline_layout));
    }
    {
      VkComputePipelineCreateInfo info;
      MEMZERO(info);
      info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      info.layout = pipeline_layout;
      info.stage  = stages[0];
      VK_ASSERT_OK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
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
    ito(limbo_items.size) {
      T &item = items[limbo_items[i].item_index];
      ((Parent_t *)this)->release_item(item);
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

struct Window {
  static constexpr u32 MAX_SC_IMAGES = 0x10;
  SDL_Window *         window;

  VkSurfaceKHR surface       = VK_NULL_HANDLE;
  i32          window_width  = 1280;
  i32          window_height = 720;

  VkInstance                 instance                 = VK_NULL_HANDLE;
  VkPhysicalDevice           physdevice               = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties device_properties        = {};
  VkQueue                    queue                    = VK_NULL_HANDLE;
  VkDevice                   device                   = VK_NULL_HANDLE;
  VkCommandPool              cmd_pools[MAX_SC_IMAGES] = {};
  VkQueryPool                query_pool               = VK_NULL_HANDLE;
  u32                        timestamp_frequency      = 0;
  u32                        query_cursor             = 0;
  // VkCommandBuffer            cmd_buffers[MAX_SC_IMAGES] = {};

  VkSwapchainKHR     swapchain                = VK_NULL_HANDLE;
  uint32_t           sc_image_count           = 0;
  ID                 sc_images[MAX_SC_IMAGES] = {};
  VkExtent2D         sc_extent                = {};
  VkSurfaceFormatKHR sc_format                = {};

  u32             frame_id                         = 0;
  u32             cmd_index                        = 0;
  u32             image_index                      = 0;
  VkFence         frame_fences[MAX_SC_IMAGES]      = {};
  VkSemaphore     sc_free_sem[MAX_SC_IMAGES]       = {};
  VkSemaphore     render_finish_sem[MAX_SC_IMAGES] = {};
  Descriptor_Pool desc_pools[MAX_SC_IMAGES]        = {};

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
      buf.release();
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
      if (img.mem_chunk_id.is_null() == false) {
        // True in case of swap chain images
        vkDestroyImage(wnd->device, img.image, NULL);
        wnd->mem_chunks[img.mem_chunk_id.index()].rem_reference();
      }
      img.release();
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
      // wnd->buffers[buf.buf_id].rem_reference();
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
      // wnd->images[img.img_id].rem_reference();
      MEMZERO(img);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } image_views;
  struct Shader_Array : Resource_Array<Shader_Info, Shader_Array> {
    static constexpr char const NAME[] = "Shader_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Shader_Info &shader) { shader.release(); }
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
  struct Compute_Pipe_Array : Resource_Array<Compute_Pipeline_Wrapper, Compute_Pipe_Array> {
    static constexpr char const NAME[] = "Compute_Pipe_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Compute_Pipeline_Wrapper &item) {
      item.release(wnd->device);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } compute_pipelines;
  struct Fence_Array : Resource_Array<Fence, struct Fence_Array> {
    static constexpr char const NAME[] = "Fence_Pipe_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Fence &item) {
      vkDestroyFence(wnd->device, item.fence, NULL);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } fences;
  struct Event_Array : Resource_Array<Event, struct Event_Array> {
    static constexpr char const NAME[] = "Event_Pipe_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Event &item) {
      vkDestroyEvent(wnd->device, item.event, NULL);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } events;
  struct CmdBuffer_Array : Resource_Array<CommandBuffer, struct CmdBuffer_Array> {
    static constexpr char const NAME[] = "CmdBuffer";
    Window *                    wnd    = NULL;
    void                        release_item(CommandBuffer &item) {
      vkFreeCommandBuffers(wnd->device, item.pool, 1, &item.cmd);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } cmd_buffers;
  struct Sem_Array : Resource_Array<Semaphore, struct Sem_Array> {
    static constexpr char const NAME[] = "Sem_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Semaphore &item) {
      vkDestroySemaphore(wnd->device, item.sem, NULL);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } semaphores;
  struct Sampler_Array : Resource_Array<Sampler, Sampler_Array> {
    static constexpr char const NAME[] = "Sampler_Array";
    Window *                    wnd    = NULL;
    void                        release_item(Sampler &item) {
      vkDestroySampler(wnd->device, item.sampler, NULL);
      MEMZERO(item);
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } samplers;
  ID                                          cur_pass;
  Hash_Table<Graphics_Pipeline_State, ID>     pipeline_cache;
  Hash_Table<ID, ID>                          compute_pipeline_cache;
  Hash_Table<u64, ID>                         shader_cache;
  Hash_Table<rd::Render_Pass_Create_Info, ID> pass_cache;

  void init_ds() {
    shader_cache.init();
    pipeline_cache.init();
    compute_pipeline_cache.init();
    mem_chunks.init();
    buffers.init(this);
    samplers.init(this);
    images.init(this);
    shaders.init(this);
    buffer_views.init(this);
    image_views.init(this);
    render_passes.init(this);
    pipelines.init(this);
    compute_pipelines.init(this);
    fences.init(this);
    events.init(this);
    cmd_buffers.init(this);
    semaphores.init(this);
  }

  void release() {
    vkDeviceWaitIdle(device);
    shader_cache.release();
    pipeline_cache.release();
    compute_pipeline_cache.release();
    buffers.release();
    samplers.release();
    images.release();
    shaders.release();
    buffer_views.release();
    image_views.release();
    render_passes.release();
    pipelines.release();
    compute_pipelines.release();
    fences.release();
    events.release();
    cmd_buffers.release();
    semaphores.release();
    ito(mem_chunks.size) mem_chunks[i].release(device);
    mem_chunks.release();
    ito(sc_image_count) vkDestroyCommandPool(device, cmd_pools[i], NULL);
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, query_pool, NULL);
    ito(sc_image_count) vkDestroySemaphore(device, sc_free_sem[i], NULL);
    ito(sc_image_count) vkDestroySemaphore(device, render_finish_sem[i], NULL);
    ito(sc_image_count) vkDestroyFence(device, frame_fences[i], NULL);
    ito(sc_image_count) desc_pools[i].release();
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroyInstance(instance, NULL);
    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  Descriptor_Pool &get_descriptor_pool() { return desc_pools[cmd_index]; }

  u32 allocate_timestamp_id() { return (query_cursor++ % 1000); }

  u32 find_mem_chunk(u32 prop_flags, u32 memory_type_bits, u32 alignment, u32 size) {
    (void)alignment;
    ito(mem_chunks.size) { // look for a suitable memory chunk
      Mem_Chunk &chunk = mem_chunks[i];
      if ((chunk.prop_flags & prop_flags) == prop_flags &&
          (chunk.memory_type_bits & memory_type_bits) == memory_type_bits) {
        if (chunk.has_space(alignment, size)) {
          return i;
        }
      }
    }
    // if failed create a new one
    Mem_Chunk new_chunk;
    u32       alloc_size = 1 << 16;
    if (alloc_size < size) {
      alloc_size = (size + alloc_size - 1) & ~(alloc_size - 1);
    }
    new_chunk.init(device, alloc_size, find_mem_type(memory_type_bits, prop_flags), prop_flags,
                   memory_type_bits);
    ito(mem_chunks.size) { // look for a free memory chunk slot
      Mem_Chunk &chunk = mem_chunks[i];
      if (chunk.is_empty()) {
        chunk = new_chunk;
        return i;
      }
    }
    ASSERT_DEBUG(new_chunk.has_space(alignment, size));
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

  Resource_ID create_buffer(rd::Buffer_Create_Info info) {
    u32                prop_flags = to_vk_memory_bits(info.mem_bits);
    VkBuffer           buf;
    VkBufferCreateInfo cinfo;
    {
      MEMZERO(cinfo);
      cinfo.sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      cinfo.pQueueFamilyIndices   = &graphics_queue_id;
      cinfo.queueFamilyIndexCount = 1;
      cinfo.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
      cinfo.size                  = info.size;
      cinfo.usage                 = to_vk_buffer_usage_bits(info.usage_bits);
      VK_ASSERT_OK(vkCreateBuffer(device, &cinfo, NULL, &buf));
    }
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buf, &reqs);
    Buffer new_buf;
    new_buf.init();
    new_buf.buffer       = buf;
    new_buf.access_flags = 0;
    new_buf.create_info  = cinfo;
    // new_buf.ref_cnt      = 1;

    u32 chunk_index = find_mem_chunk(prop_flags, reqs.memoryTypeBits, reqs.alignment, reqs.size);
    new_buf.mem_chunk_id = ID{chunk_index + 1};
    Mem_Chunk &chunk     = mem_chunks[chunk_index];
    new_buf.mem_offset   = chunk.alloc(reqs.alignment, reqs.size);

    vkBindBufferMemory(device, new_buf.buffer, chunk.mem, new_buf.mem_offset);

    return {buffers.push(new_buf), (i32)Resource_Type::BUFFER};
  }

  void *map_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)Resource_Type::BUFFER);
    Buffer &   buf   = buffers[res_id.id];
    Mem_Chunk &chunk = mem_chunks[buf.mem_chunk_id.index()];
    void *     data  = NULL;
    VK_ASSERT_OK(vkMapMemory(device, chunk.mem, buf.mem_offset, buf.create_info.size, 0, &data));
    return data;
  }

  void unmap_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)Resource_Type::BUFFER);
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

  VkCommandPool cur_cmd_pool() { return cmd_pools[cmd_index]; }

  Resource_ID create_fence(bool signaled) {
    VkFenceCreateInfo info;
    MEMZERO(info);
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    Fence fence;
    fence.init();
    VK_ASSERT_OK(vkCreateFence(device, &info, NULL, &fence.fence));
    ID fence_id = fences.push(fence);
    return {fence_id, (u32)Resource_Type::FENCE};
  }

  Resource_ID create_event() {
    VkEventCreateInfo info;
    MEMZERO(info);
    info.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    Event event;
    event.init();
    VK_ASSERT_OK(vkCreateEvent(device, &info, NULL, &event.event));
    ID fence_id = events.push(event);
    return {fence_id, (u32)Resource_Type::EVENT};
  }

  Resource_ID create_command_buffer() {
    CommandBuffer cmd;
    MEMZERO(cmd);
    cmd.pool = cmd_pools[cmd_index];

    VkCommandBufferAllocateInfo info;
    MEMZERO(info);
    info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool        = cmd_pools[cmd_index];
    info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VK_ASSERT_OK(vkAllocateCommandBuffers(device, &info, &cmd.cmd));
    ID id = cmd_buffers.push(cmd);
    return {id, (u32)Resource_Type::COMMAND_BUFFER};
  }

  Resource_ID create_semaphore() {
    VkSemaphoreCreateInfo info;
    MEMZERO(info);
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = NULL;
    info.flags = 0;
    Semaphore s;
    s.init();
    vkCreateSemaphore(device, &info, 0, &s.sem);
    ID id = semaphores.push(s);
    return {id, (u32)Resource_Type::SEMAPHORE};
  }

  Resource_ID create_image_view(ID res_id, u32 base_level, u32 levels, u32 base_layer, u32 layers,
                                VkFormat format) {
    Image &   img = images[res_id];
    ImageView img_view;
    MEMZERO(img_view);
    VkImageViewCreateInfo cinfo;
    MEMZERO(cinfo);
    cinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    VkComponentMapping cm;
    cm.r             = VK_COMPONENT_SWIZZLE_R;
    cm.g             = VK_COMPONENT_SWIZZLE_G;
    cm.b             = VK_COMPONENT_SWIZZLE_B;
    cm.a             = VK_COMPONENT_SWIZZLE_A;
    cinfo.components = cm;
    if (format == VK_FORMAT_UNDEFINED)
      cinfo.format = img.info.format;
    else
      cinfo.format = format;
    cinfo.image                           = img.image;
    cinfo.subresourceRange.aspectMask     = img.aspect;
    cinfo.subresourceRange.baseArrayLayer = base_layer;
    cinfo.subresourceRange.baseMipLevel   = base_level;
    cinfo.subresourceRange.layerCount     = layers;
    cinfo.subresourceRange.levelCount     = levels;
    cinfo.viewType =
        img.info.extent.depth == 1
            ? (img.info.extent.height == 1 ? //
                   (img.info.arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_1D_ARRAY)
                                           : //
                   (img.info.arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_2D
                                              : VK_IMAGE_VIEW_TYPE_2D_ARRAY))
            : VK_IMAGE_VIEW_TYPE_3D;

    img_view.flags.components       = cinfo.components;
    img_view.flags.format           = cinfo.format;
    img_view.flags.subresourceRange = cinfo.subresourceRange;
    img_view.flags.viewType         = cinfo.viewType;
    // check if there's already a view with needed properties
    ito(img.views.size) {
      ImageView &view = image_views[img.views[i]];
      if (view.flags == img_view.flags) {
        return {img.views[i], (u32)Resource_Type::IMAGE_VIEW};
      }
    }

    VK_ASSERT_OK(vkCreateImageView(device, &cinfo, NULL, &img_view.view));
    img_view.img_id = res_id;
    // img.add_reference();
    ID view_id = image_views.push(img_view);
    img.views.push(view_id);
    return {view_id, (i32)Resource_Type::IMAGE_VIEW};
  }

  Resource_ID create_image(u32 width, u32 height, u32 depth, u32 layers, u32 levels,
                           VkFormat format, u32 usage_flags, u32 mem_flags) {
    u32               prop_flags = to_vk_memory_bits(mem_flags);
    VkImage           image;
    VkImageCreateInfo cinfo;
    {
      MEMZERO(cinfo);
      cinfo.flags                 = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
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
      /*if ((usage_flags & (u32)rd::Image_Usage_Bits::USAGE_RT) != 0 ||
          (usage_flags & (u32)rd::Image_Usage_Bits::USAGE_DT) != 0)*/
      cinfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      /* else
         cinfo.tiling = VK_IMAGE_TILING_LINEAR;*/
      cinfo.usage = to_vk_image_usage_bits(usage_flags);
      VK_ASSERT_OK(vkCreateImage(device, &cinfo, NULL, &image));
    }
    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(device, image, &reqs);
    Image new_image;
    new_image.init();
    new_image.image              = image;
    new_image.access_flags       = 0;
    new_image.info.arrayLayers   = cinfo.arrayLayers;
    new_image.info.extent.width  = cinfo.extent.width;
    new_image.info.extent.height = cinfo.extent.height;
    new_image.info.extent.depth  = cinfo.extent.depth;
    new_image.info.format        = cinfo.format;
    new_image.info.imageType     = cinfo.imageType;
    new_image.info.mipLevels     = cinfo.mipLevels;
    new_image.info.samples       = cinfo.samples;
    new_image.info.sharingMode   = cinfo.sharingMode;
    new_image.info.tiling        = cinfo.tiling;
    new_image.info.usage         = cinfo.usage;
    new_image.layout             = VK_IMAGE_LAYOUT_UNDEFINED;
    // new_image.ref_cnt            = 1;
    if (usage_flags & (u32)rd::Image_Usage_Bits::USAGE_DT)
      new_image.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else
      new_image.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    u32 chunk_index = find_mem_chunk(prop_flags, reqs.memoryTypeBits, reqs.alignment, reqs.size);
    new_image.mem_chunk_id = ID{chunk_index + 1};
    Mem_Chunk &chunk       = mem_chunks[chunk_index];
    new_image.mem_offset   = chunk.alloc(reqs.alignment, reqs.size);
    vkBindImageMemory(device, new_image.image, chunk.mem, new_image.mem_offset);
    ID          img_id = images.push(new_image);
    Resource_ID res_id = {img_id, (u32)Resource_Type::IMAGE};
    return res_id;
  }

  Resource_ID create_shader_raw(rd::Stage_t type, string_ref body,
                                Pair<string_ref, string_ref> *defines, size_t num_defines) {
    u64 shader_hash = hash_of(body);
    ito(num_defines) { shader_hash ^= hash_of(defines[0].first) ^ hash_of(defines[0].second); }
    if (shader_cache.contains(shader_hash)) {
      return {shader_cache.get(shader_hash), (u32)Resource_Type::SHADER};
    }

    String_Builder sb;
    sb.init();
    defer(sb.release());
    sb.reset();
    preprocess_shader(sb, body);
    string_ref text = sb.get_str();

    Shader_Info         si;
    shaderc_shader_kind kind;
    if (type == rd::Stage_t::VERTEX)
      kind = shaderc_vertex_shader;
    else if (type == rd::Stage_t::COMPUTE)
      kind = shaderc_compute_shader;
    else if (type == rd::Stage_t::PIXEL)
      kind = shaderc_fragment_shader;
    else
      UNIMPLEMENTED;

    si.init(type, shader_hash, compile_glsl(device, text, kind, defines, num_defines));

    ID shid = shaders.push(si);
    shader_cache.insert(shader_hash, shid);
    return {shid, (u32)Resource_Type::SHADER};
  }
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) {
    Sampler sm;
    sm.create_info = info;
    VkSamplerCreateInfo cinfo;
    MEMZERO(cinfo);
    cinfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    cinfo.addressModeU            = to_vk(info.address_mode_u);
    cinfo.addressModeV            = to_vk(info.address_mode_v);
    cinfo.addressModeW            = to_vk(info.address_mode_w);
    cinfo.anisotropyEnable        = to_vk(info.anisotropy);
    cinfo.compareEnable           = to_vk(info.cmp);
    cinfo.compareOp               = to_vk(info.cmp_op);
    cinfo.magFilter               = to_vk(info.mag_filter);
    cinfo.minFilter               = to_vk(info.min_filter);
    cinfo.maxAnisotropy           = info.max_anisotropy;
    cinfo.minLod                  = info.min_lod;
    cinfo.maxLod                  = info.max_lod;
    cinfo.unnormalizedCoordinates = to_vk(info.unnormalized_coordiantes);
    cinfo.mipLodBias              = info.mip_lod_bias;
    cinfo.mipmapMode = info.mip_mode == rd::Filter::LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                           : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VK_ASSERT_OK(vkCreateSampler(device, &cinfo, NULL, &sm.sampler));
    ID id = samplers.push(sm);
    return {id, (u32)Resource_Type::SAMPLER};
  }
  rd::Image2D_Info get_swapchain_image_info() {
    rd::Image2D_Info info;
    MEMZERO(info);
    Image &img  = images[sc_images[image_index]];
    info.format = from_vk(img.info.format);
    info.height = img.info.extent.height;
    info.width  = img.info.extent.width;
    info.layers = img.info.arrayLayers;
    info.levels = img.info.mipLevels;
    return info;
  }
  rd::Image_Info get_image_info(Resource_ID res_id) {
    rd::Image_Info info;
    MEMZERO(info);
    Image &img    = images[res_id.id];
    info.format   = from_vk(img.info.format);
    info.height   = img.info.extent.height;
    info.depth    = img.info.extent.depth;
    info.width    = img.info.extent.width;
    info.layers   = img.info.arrayLayers;
    info.levels   = img.info.mipLevels;
    info.is_depth = img.is_depth_image();
    return info;
  }

  void release_resource(Resource_ID res_id) {
    if (res_id.type == (u32)Resource_Type::PASS) {
      Render_Pass &pass = render_passes[res_id.id];
      render_passes.remove(res_id.id, 3);
      if (pass_cache.contains(pass.create_info)) pass_cache.remove(pass.create_info);
    } else if (res_id.type == (u32)Resource_Type::BUFFER) {
      Buffer &buf = buffers[res_id.id];
      // buf.rem_reference();
      ito(buf.views.size) buffer_views.remove(buf.views[i], 3);
      buffers.remove(res_id.id, 4);
    } else if (res_id.type == (u32)Resource_Type::BUFFER_VIEW) {
      BufferView &view = buffer_views[res_id.id];
      Buffer &    buf  = buffers[view.buf_id];
      buf.views.remove(res_id.id);
      buffer_views.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::IMAGE_VIEW) {
      ImageView &view = image_views[res_id.id];
      Image &    img  = images[view.img_id];
      img.views.remove(res_id.id);
      image_views.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::IMAGE) {
      Image &img = images[res_id.id];
      images.remove(res_id.id, 3);
      // img.rem_reference();
      ito(img.views.size) image_views.remove(img.views[i], 3);
    } else if (res_id.type == (u32)Resource_Type::SHADER) {
      shaders.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::FENCE) {
      fences.remove(res_id.id, 4);
    } else if (res_id.type == (u32)Resource_Type::EVENT) {
      events.remove(res_id.id, 4);
    } else if (res_id.type == (u32)Resource_Type::SEMAPHORE) {
      semaphores.remove(res_id.id, 4);
    } else if (res_id.type == (u32)Resource_Type::COMMAND_BUFFER) {
      cmd_buffers.remove(res_id.id, 4);
    } else if (res_id.type == (u32)Resource_Type::SAMPLER) {
      samplers.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::TIMESTAMP) {
    } else {
      TRAP;
    }
  }

  void release_swapchain() {
    if (swapchain != VK_NULL_HANDLE) {
      vkDestroySwapchainKHR(device, swapchain, NULL);
    }
    ito(sc_image_count) {
      Image &img = images[sc_images[i]];
      // img.rem_reference();
      jto(img.views.size) { image_views.remove(img.views[j], 3); }
      images.remove(sc_images[i], 3);
      sc_images[i] = ID{0};
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
      // if (present_modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) { // prefer mailbox
      // present_mode_of_choice = VK_PRESENT_MODE_IMMEDIATE_KHR;
      // break;
      //}
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
    sc_create_info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    sc_create_info.preTransform          = surface_capabilities.currentTransform;
    sc_create_info.compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc_create_info.presentMode           = present_mode_of_choice;
    sc_create_info.clipped               = VK_TRUE;
    sc_create_info.queueFamilyIndexCount = 1;
    sc_create_info.pQueueFamilyIndices   = &graphics_queue_id;

    sc_image_count = 0;
    VK_ASSERT_OK(vkCreateSwapchainKHR(device, &sc_create_info, 0, &swapchain));
    vkGetSwapchainImagesKHR(device, swapchain, &sc_image_count, NULL);
    VkImage raw_images[MAX_SC_IMAGES];
    vkGetSwapchainImagesKHR(device, swapchain, &sc_image_count, raw_images);
    ito(sc_image_count) {
      Image image;
      image.init();
      image.image              = raw_images[i];
      image.access_flags       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      image.layout             = VK_IMAGE_LAYOUT_UNDEFINED;
      image.info.arrayLayers   = sc_create_info.imageArrayLayers;
      image.info.extent.width  = sc_create_info.imageExtent.width;
      image.info.extent.height = sc_create_info.imageExtent.height;
      image.info.extent.depth  = 1;
      image.info.format        = sc_create_info.imageFormat;
      image.info.imageType     = VK_IMAGE_TYPE_2D;
      image.info.mipLevels     = 1;
      image.info.samples       = VK_SAMPLE_COUNT_1_BIT;
      image.info.sharingMode   = sc_create_info.imageSharingMode;
      image.info.tiling        = VK_IMAGE_TILING_OPTIMAL;
      image.info.usage         = sc_create_info.imageUsage;
      image.aspect             = VK_IMAGE_ASPECT_COLOR_BIT;
      // image.ref_cnt            = 1;
      sc_images[i] = images.push(image);
    }
  }

  void init() {
    init_ds();
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 512, 512,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
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

    const char *layerNames[]      = {"VK_LAYER_KHRONOS_validation"};
    bool        enable_validation = false;
#ifndef NDEBUG
    enable_validation = true;
#endif
    VkInstanceCreateInfo info;
    MEMZERO(info);
    info.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app_info;
    if (enable_validation) {
      info.enabledLayerCount   = ARRAY_SIZE(layerNames);
      info.ppEnabledLayerNames = layerNames;
    } else {
      info.enabledLayerCount = 0;
    }
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
        VkPhysicalDeviceProperties Properties;
        vkGetPhysicalDeviceProperties(physdevice_handles[i], &Properties);
        if (Properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
          if (graphics_device_id != NULL)
            break; // Just stop right there if we've found a discrete gpu
        }
      }
    }
    physdevice                             = graphics_device_id;
    char const *       device_extensions[] = {//
                                       VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                       // VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
                                       // VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                                       VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo device_create_info;
    MEMZERO(device_create_info);
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    if (enable_validation) {
      device_create_info.enabledLayerCount   = ARRAY_SIZE(layerNames);
      device_create_info.ppEnabledLayerNames = layerNames;
    } else {
      device_create_info.enabledLayerCount = 0;
    }
    device_create_info.enabledExtensionCount   = ARRAY_SIZE(device_extensions);
    device_create_info.ppEnabledExtensionNames = device_extensions;
    float                   priority           = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info;
    MEMZERO(queue_create_info);
    queue_create_info.sType                 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex      = graphics_queue_id;
    queue_create_info.queueCount            = 1;
    queue_create_info.pQueuePriorities      = &priority;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos    = &queue_create_info;

    VkPhysicalDeviceFeatures2 pd_features2;
    MEMZERO(pd_features2);
    pd_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT pd_index_features;
    MEMZERO(pd_index_features);
    pd_index_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;

    pd_features2.pNext = ((void *)&pd_index_features);
    vkGetPhysicalDeviceFeatures2(physdevice, &pd_features2);
    ASSERT_DEBUG(pd_index_features.shaderSampledImageArrayNonUniformIndexing);
    ASSERT_DEBUG(pd_index_features.descriptorBindingPartiallyBound);
    ASSERT_DEBUG(pd_index_features.runtimeDescriptorArray);

    VkPhysicalDeviceFeatures pd_features;
    MEMZERO(pd_features);
    vkGetPhysicalDeviceFeatures(physdevice, &pd_features);
    ASSERT_DEBUG(pd_features.fillModeNonSolid);

    VkPhysicalDeviceHostQueryResetFeatures query_reset_features;
    MEMZERO(query_reset_features);
    query_reset_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT;
    query_reset_features.hostQueryReset = VK_TRUE;

    VkPhysicalDeviceScalarBlockLayoutFeatures scalar_features;
    MEMZERO(scalar_features);
    scalar_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
    scalar_features.scalarBlockLayout = VK_TRUE;

    query_reset_features.pNext = &scalar_features;
    pd_index_features.pNext    = &query_reset_features;

    device_create_info.pNext = &pd_index_features;

    VkPhysicalDeviceFeatures enabled_features;
    MEMZERO(enabled_features);
    enabled_features                                         = pd_features;
    enabled_features.multiDrawIndirect                       = VK_TRUE;
    enabled_features.shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
    enabled_features.shaderSampledImageArrayDynamicIndexing  = VK_TRUE;
    enabled_features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    enabled_features.shaderStorageImageArrayDynamicIndexing  = VK_TRUE;
    device_create_info.pEnabledFeatures                      = &enabled_features;

    VK_ASSERT_OK(vkCreateDevice(graphics_device_id, &device_create_info, NULL, &device));
    vkGetDeviceQueue(device, graphics_queue_id, 0, &queue);
    ASSERT_ALWAYS(queue != VK_NULL_HANDLE);
    vkGetPhysicalDeviceProperties(physdevice, &device_properties);
    update_swapchain();
    {
      VkCommandPoolCreateInfo info;
      MEMZERO(info);
      info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      info.queueFamilyIndex = graphics_queue_id;

      ito(sc_image_count) VK_ASSERT_OK(vkCreateCommandPool(device, &info, 0, &cmd_pools[i]));
    }
    {
      VkQueryPoolCreateInfo info;
      MEMZERO(info);
      info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
      info.queryCount = 1000;
      VK_ASSERT_OK(vkCreateQueryPool(device, &info, NULL, &query_pool));
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
    ito(sc_image_count) desc_pools[i].init(device);
  }

  void update_surface_size() {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdevice, surface, &surface_capabilities);
    window_width  = surface_capabilities.currentExtent.width;
    window_height = surface_capabilities.currentExtent.height;
  }

  void start_frame() {
    buffers.tick();
    samplers.tick();
    images.tick();
    shaders.tick();
    buffer_views.tick();
    image_views.tick();
    render_passes.tick();
    pipelines.tick();
    compute_pipelines.tick();
    fences.tick();
    events.tick();
    cmd_buffers.tick();
    semaphores.tick();
    {
      ito(mem_chunks.size) {
        Mem_Chunk &chunk = mem_chunks[i];
        if (chunk.is_referenced() == false) {
          chunk.release(device);
        }
      }
    }
    Array<Resource_ID> resources_to_remove;
    resources_to_remove.init();
    defer(resources_to_remove.release());
    render_passes.for_each([&](Render_Pass const &p) {
      if (p.frames_referenced < -2) {
        resources_to_remove.push({p.id, (u32)Resource_Type::PASS});
      }
    });
    ito(resources_to_remove.size) release_resource(resources_to_remove[i]);
  restart:
    update_surface_size();
    if (window_width != (i32)sc_extent.width || window_height != (i32)sc_extent.height) {
      update_swapchain();
    }
    cmd_index = (frame_id++) % sc_image_count;

    VkResult wait_res = vkWaitForFences(device, 1, &frame_fences[cmd_index], VK_TRUE, 1000);
    if (wait_res == VK_TIMEOUT) {
      goto restart;
    }
    vkResetFences(device, 1, &frame_fences[cmd_index]);
    desc_pools[cmd_index].reset();
    VkResult acquire_res = vkAcquireNextImageKHR(
        device, swapchain, UINT64_MAX, sc_free_sem[cmd_index], VK_NULL_HANDLE, &image_index);

    if (acquire_res == VK_ERROR_OUT_OF_DATE_KHR || acquire_res == VK_SUBOPTIMAL_KHR) {
      update_swapchain();
      goto restart;
    } else if (acquire_res != VK_SUCCESS) {
      TRAP;
    }
    VK_ASSERT_OK(vkResetCommandPool(device, cmd_pools[cmd_index],
                                    VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
  }
  void end_frame(VkSemaphore *wait_sem) {
    Resource_ID   cmd_id = create_command_buffer();
    CommandBuffer cmd    = cmd_buffers[cmd_id.id];
    // vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    VkCommandBufferBeginInfo begin_info;
    MEMZERO(begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd.cmd, &begin_info);
    Image &img = images[sc_images[image_index]];
    img.barrier(cmd.cmd, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    vkEndCommandBuffer(cmd.cmd);
    release_resource(cmd_id);
    VkPipelineStageFlags stage_flags[]{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    VkSubmitInfo         submit_info;
    MEMZERO(submit_info);
    if (wait_sem != NULL) {
      VkSemaphore sems[2]              = {sc_free_sem[cmd_index], *wait_sem};
      submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.waitSemaphoreCount   = 2;
      submit_info.pWaitSemaphores      = sems;
      submit_info.pWaitDstStageMask    = stage_flags;
      submit_info.commandBufferCount   = 1;
      submit_info.pCommandBuffers      = &cmd.cmd;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores    = &render_finish_sem[cmd_index];
      vkQueueSubmit(queue, 1, &submit_info, frame_fences[cmd_index]);
    } else {
      submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.waitSemaphoreCount   = 1;
      submit_info.pWaitSemaphores      = &sc_free_sem[cmd_index];
      submit_info.pWaitDstStageMask    = stage_flags;
      submit_info.commandBufferCount   = 1;
      submit_info.pCommandBuffers      = &cmd.cmd;
      submit_info.signalSemaphoreCount = 1;
      submit_info.pSignalSemaphores    = &render_finish_sem[cmd_index];
      vkQueueSubmit(queue, 1, &submit_info, frame_fences[cmd_index]);
    }
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

class Vk_Ctx : public rd::Imm_Ctx {
  private:
  rd::Pass_t                              type;
  Window *                                wnd;
  Graphics_Pipeline_State                 graphics_state;
  ID                                      cs;
  Resource_ID                             cmd_id;
  VkCommandBuffer                         cmd;
  InlineArray<Graphics_Pipeline_State, 8> stack;
  ID                                      cur_pass;

  enum class Binding_t { UNIFORM_BUFFER, STORAGE_BUFFER, IMAGE, SAMPLER, STORAGE_IMAGE };
  struct Resource_Binding {
    u32       set;
    u32       binding;
    u32       dummy;
    u32       element;
    Binding_t type;
    union {
      struct {
        ID     buf_id;
        size_t offset;
        size_t size;
      } uniform_buffer;
      struct {
        ID     buf_id;
        size_t offset;
        size_t size;
      } storage_buffer;
      struct {
        ID                      image_id;
        VkImageSubresourceRange sr;
        VkFormat                format;
      } image;
      struct {
        ID sampler_id;
      } sampler;
    };
    void reset() { memset(this, 0, sizeof(*this)); }
    bool operator==(Resource_Binding const &that) const {
      return memcmp(this, &that, sizeof(*this)) == 0;
    }
  };

  template <typename K, typename V> using Table = Hash_Table<K, V, Default_Allocator, 0x1000>;
  Table<Resource_Path, Resource_Binding> deferred_bindings;
  VkDescriptorSet                        set_cache[0x10];
  u64                                    set_dirty_flags;

  struct VBO_Binding {
    VkBuffer     buffer;
    VkDeviceSize offset;
  };

  struct IBO_Binding {
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkIndexType  indexType;
  };

  enum class Cmd_t : u8 {
    DRAW = 0,
    PUSH_CONSTANTS,
    DISPATCH,
    BARRIER,
    RENDER_RECT,
    BUFFER_FILL,
    CLEAR_IMAGE,
    TIMESTAMP,
    COPY_BUFFER_IMAGE,
    COPY_IMAGE_BUFFER,
    COPY_BUFFER,
    EVENT
  };

  struct Deferred_Push_Constants {
    u32 offset;
    u32 size;
  };

  struct Deferred_Rect {
    bool is_viewport;
    union {
      VkViewport viewport;
      VkRect2D   scissor;
    };
  };

  struct Deferred_Draw {
    enum class Type : u32 { UNKNOWN = 0, DRAW, DRAW_INDEXED, MULTI_DRAW_INDEXED_INDIRECT };
    ID                              pso;
    InlineArray<VkDescriptorSet, 8> sets;
    InlineArray<VBO_Binding, 8>     vbos;
    IBO_Binding                     ibo;
    Type                            type;
    float                           depth_bias;
    float                           line_width;
    union {
      struct {
        u32 index_count;
        u32 instance_count;
        u32 first_index;
        u32 first_instance;
        i32 vertex_offset;
      } draw_indexed;
      struct {
        u32 vertex_count;
        u32 instance_count;
        u32 first_vertex;
        u32 first_instance;
      } draw;
      struct {
        ID  arg_buf_id;
        u32 arg_buf_offset;
        ID  cnt_buf_id;
        u32 cnt_buf_offset;
        u32 max_count;
        u32 stride;
      } multi_draw_indexed_indirect;
    };
    void reset() {
      memset(this, 0, sizeof(*this));
      line_width = 1.0f;
    }
  };

  struct Deferred_Dispatch {
    // enum class Type : u32 { UNKNOWN = 0, DISPATCH, DISPATCH_INDIRECT };
    // Type                            type;
    ID                              pso;
    InlineArray<VkDescriptorSet, 8> sets;
    union {
      struct {
        u32 x, y, z;
      } dispatch;
    };
    void reset() { memset(this, 0, sizeof(*this)); }
  };

  struct Deferred_Barrier {
    bool          image;
    ID            res_id;
    VkImageLayout new_layout;
    VkAccessFlags new_access_flags;
    void          init() { memset(this, 0, sizeof(*this)); }
    void          release() {}
  };

  struct Deferred_Buffer_Fill {
    ID     buf_id;
    size_t offset;
    size_t size;
    u32    val;
  };

  struct Deferred_Clear_Image {
    ID                      img_id;
    VkClearColorValue       cv;
    VkImageSubresourceRange r;
  };

  struct Deferred_Copy_Buffer_Image {
    Resource_ID    buf_id;
    size_t         offset;
    Resource_ID    img_id;
    rd::Image_Copy dst_info;
  };

  struct Deferred_Event {
    VkEvent event;
  };

  struct Deferred_Copy_Buffer {
    Resource_ID src_buf_id;
    size_t      src_offset;
    Resource_ID dst_buf_id;
    size_t      dst_offset;
    u32         size;
  };

  struct Deferred_Timestamp {
    VkQueryPool pool;
    u32         timestamp_id;
  };

  void _copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                    size_t dst_offset, u32 size) {
    Buffer &src_buffer = wnd->buffers[src_buf_id.id];
    Buffer &dst_buffer = wnd->buffers[dst_buf_id.id];

    src_buffer.barrier(cmd, VK_ACCESS_TRANSFER_READ_BIT);
    dst_buffer.barrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferCopy info;
    MEMZERO(info);
    info.dstOffset = dst_offset;
    info.size      = size;
    info.srcOffset = src_offset;
    vkCmdCopyBuffer(cmd, src_buffer.buffer, dst_buffer.buffer, 1, &info);
  }

  void _copy_image_buffer(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                          rd::Image_Copy const &dst_info) {
    Buffer &buffer           = wnd->buffers[buf_id.id];
    Image & image            = wnd->images[img_id.id];
    auto    old_access_flags = image.access_flags;
    auto    old_layout       = image.layout;

    image.barrier(cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy info;
    MEMZERO(info);
    info.bufferOffset = offset;
    // info.bufferRowLength = 0; // image.getbpp() * image.info.extent.width;
    if (dst_info.size_x == 0)
      info.imageExtent.width = image.info.extent.width;
    else
      info.imageExtent.width = dst_info.size_x;

    if (dst_info.size_y == 0)
      info.imageExtent.height = image.info.extent.height;
    else
      info.imageExtent.height = dst_info.size_y;

    if (dst_info.size_z == 0)
      info.imageExtent.depth = image.info.extent.depth;
    else
      info.imageExtent.depth = dst_info.size_z;

    info.imageOffset = {(i32)dst_info.offset_x, (i32)dst_info.offset_y, (i32)dst_info.offset_z};
    VkImageSubresourceLayers subres;
    MEMZERO(subres);
    subres.aspectMask     = image.aspect;
    subres.baseArrayLayer = dst_info.layer;
    if (dst_info.num_layers == 0)
      subres.layerCount = 1;
    else
      subres.layerCount = dst_info.num_layers;
    subres.mipLevel        = dst_info.level;
    info.imageSubresource  = subres;
    info.bufferImageHeight = image.info.extent.height;
    vkCmdCopyImageToBuffer(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1,
                           &info);
  }

  void _copy_buffer_image(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                          rd::Image_Copy const &dst_info) {
    Buffer &buffer           = wnd->buffers[buf_id.id];
    Image & image            = wnd->images[img_id.id];
    auto    old_access_flags = image.access_flags;
    auto    old_layout       = image.layout;

    image.barrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy info;
    MEMZERO(info);
    info.bufferOffset = offset;
    // info.bufferRowLength = 0; // image.getbpp() * image.info.extent.width;
    if (dst_info.size_x == 0)
      info.imageExtent.width = image.info.extent.width;
    else
      info.imageExtent.width = dst_info.size_x;

    if (dst_info.size_y == 0)
      info.imageExtent.height = image.info.extent.height;
    else
      info.imageExtent.height = dst_info.size_y;

    if (dst_info.size_z == 0)
      info.imageExtent.depth = image.info.extent.depth;
    else
      info.imageExtent.depth = dst_info.size_z;

    info.imageOffset = {(i32)dst_info.offset_x, (i32)dst_info.offset_y, (i32)dst_info.offset_z};
    VkImageSubresourceLayers subres;
    MEMZERO(subres);
    subres.aspectMask     = image.aspect;
    subres.baseArrayLayer = dst_info.layer;
    if (dst_info.num_layers == 0)
      subres.layerCount = 1;
    else
      subres.layerCount = dst_info.num_layers;
    subres.mipLevel        = dst_info.level;
    info.imageSubresource  = subres;
    info.bufferImageHeight = image.info.extent.height;
    vkCmdCopyBufferToImage(cmd, buffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &info);
  }

  struct CPU_Command_Buffer {
    Array<u8> data;
    size_t    cursor;
    void      init() {
      data.init();
      cursor = 0;
    }
    void release() { data.release(); }
    void reset() {
      cursor = 0;
      data.reset();
    }
    void  restart() { cursor = 0; }
    Cmd_t read_cmd_type() {
      u8 byte = data[cursor++];
      return (Cmd_t)byte;
    }
    template <typename T> void write(Cmd_t type, T const *src) {
      data.push((u8)type);
      size_t dst_pos = data.size;
      data.resize(data.size + sizeof(T));
      memcpy(data.ptr + dst_pos, src, sizeof(T));
    }
    template <typename T> void read(T *dst) {
      memcpy(dst, data.ptr + cursor, sizeof(T));
      cursor += sizeof(T);
      ASSERT_ALWAYS(cursor <= data.size);
    }
    void write(void const *src, size_t size) {
      size_t dst_pos = data.size;
      data.resize(data.size + size);
      memcpy(data.ptr + dst_pos, src, size);
    }
    void *read(size_t size) {
      void *out = data.ptr + cursor;
      cursor += size;
      ASSERT_ALWAYS(cursor <= data.size);
      return out;
    }
    bool has_data() { return cursor < data.size; }
  };
  CPU_Command_Buffer cpu_cmd;
  Deferred_Draw      current_draw;
  Deferred_Dispatch  current_dispatch;

  void push_buffer_barrier(ID buffer_id, VkAccessFlags new_access_flags) {
    Deferred_Barrier db;
    db.init();
    db.image            = false;
    db.res_id           = buffer_id;
    db.new_access_flags = new_access_flags;
    cpu_cmd.write(Cmd_t::BARRIER, &db);
  }

  void push_image_barrier(ID image_id, VkAccessFlags new_access_flags, VkImageLayout new_layout) {
    Deferred_Barrier db;
    db.init();
    db.image            = true;
    db.res_id           = image_id;
    db.new_layout       = new_layout;
    db.new_access_flags = new_access_flags;
    cpu_cmd.write(Cmd_t::BARRIER, &db);
  }
  void flush_render_pass() {
    Render_Pass &pass = wnd->render_passes[cur_pass];
    ito(pass.create_info.rts.size) {
      Image &img = wnd->images[pass.create_info.rts[i].image.id];
      img.barrier(cmd, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    if (pass.depth_target_view.is_null() == false) {
      Image &img = wnd->images[pass.create_info.depth_target.image.id];
      img.barrier(cmd, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
    VkRenderPassBeginInfo binfo;
    MEMZERO(binfo);
    binfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    binfo.clearValueCount = 0;
    binfo.framebuffer     = pass.fb;
    binfo.renderArea      = VkRect2D{{0, 0}, {pass.width, pass.height}};
    binfo.renderPass      = pass.pass;
    binfo.pClearValues    = &pass.clear_values[0];
    binfo.clearValueCount = pass.clear_values.size;
    /*{
      u64      timestamp_results[2];
      VkResult res = vkGetQueryPoolResults(wnd->device, wnd->query_pool, timestamp_begin_id, 2, 16,
                                           timestamp_results, 8, VK_QUERY_RESULT_64_BIT);
      if (res == VK_SUCCESS) {
        u64 diff = timestamp_results[1] - timestamp_results[0];
        last_ms  = diff * wnd->device_properties.limits.timestampPeriod;
      }
    }
    vkResetQueryPool(wnd->device, wnd->query_pool, timestamp_begin_id, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, wnd->query_pool,
                        timestamp_begin_id);*/
    vkCmdBeginRenderPass(cmd, &binfo, VK_SUBPASS_CONTENTS_INLINE);
    u8    pcs[128];
    u32   pc_dirty_range = 0;
    float line_width     = 1.0f;
    float depth_bias     = 0.0f;
    vkCmdSetDepthBias(cmd, depth_bias, 0.0f, 0.0f);
    vkCmdSetLineWidth(cmd, line_width);
    InlineArray<VkEvent, 16> deferred_events;
    deferred_events.init();
    while (cpu_cmd.has_data()) {
      Cmd_t type = cpu_cmd.read_cmd_type();
      if (type == Cmd_t::TIMESTAMP) {
        Deferred_Timestamp dt;
        cpu_cmd.read(&dt);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dt.pool, dt.timestamp_id);
      } else if (type == Cmd_t::PUSH_CONSTANTS) {
        Deferred_Push_Constants dpc;
        cpu_cmd.read(&dpc);
        memcpy(pcs + dpc.offset, cpu_cmd.read(dpc.size), dpc.size);
        pc_dirty_range = dpc.offset + dpc.size;
      } else if (type == Cmd_t::RENDER_RECT) {
        Deferred_Rect dr;
        cpu_cmd.read(&dr);
        if (dr.is_viewport) {
          vkCmdSetViewport(cmd, 0, 1, &dr.viewport);
        } else {
          vkCmdSetScissor(cmd, 0, 1, &dr.scissor);
        }
      } else if (type == Cmd_t::DRAW) {
        Deferred_Draw dd;
        cpu_cmd.read(&dd);
        Graphics_Pipeline_Wrapper &gw = wnd->pipelines[dd.pso];
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gw.pipeline);
        if (dd.sets.size > 0)
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gw.pipeline_layout, 0,
                                  dd.sets.size, &dd.sets[0], 0, NULL);
        if (depth_bias != dd.depth_bias) {
          vkCmdSetDepthBias(cmd, dd.depth_bias, 1.0f, 0.0f);
          depth_bias = dd.depth_bias;
        }
        if (line_width != 0.0f && line_width != dd.line_width) {
          vkCmdSetLineWidth(cmd, dd.line_width);
          line_width = dd.line_width;
        }
        jto(dd.vbos.size) {
          vkCmdBindVertexBuffers(cmd, j, 1, &dd.vbos[j].buffer, &dd.vbos[j].offset);
        }
        if (pc_dirty_range != 0) {
          if (gw.push_constants_size != 0) {
            vkCmdPushConstants(cmd, gw.pipeline_layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0,
                               MIN(pc_dirty_range, gw.push_constants_size), pcs);
          }
          pc_dirty_range = 0;
        }
        switch (dd.type) {
        case Deferred_Draw::Type::DRAW: {
          vkCmdDraw(cmd, dd.draw.vertex_count, dd.draw.instance_count, dd.draw.first_vertex,
                    dd.draw.first_instance);
          break;
        }
        case Deferred_Draw::Type::DRAW_INDEXED: {
          vkCmdBindIndexBuffer(cmd, dd.ibo.buffer, dd.ibo.offset, dd.ibo.indexType);
          vkCmdDrawIndexed(cmd, dd.draw_indexed.index_count, dd.draw_indexed.instance_count,
                           dd.draw_indexed.first_index, dd.draw_indexed.vertex_offset,
                           dd.draw_indexed.first_instance);
          break;
        }
        case Deferred_Draw::Type::MULTI_DRAW_INDEXED_INDIRECT: {
          Buffer &arg_buf = wnd->buffers[dd.multi_draw_indexed_indirect.arg_buf_id];
          Buffer &cnt_buf = wnd->buffers[dd.multi_draw_indexed_indirect.cnt_buf_id];
          vkCmdBindIndexBuffer(cmd, dd.ibo.buffer, dd.ibo.offset, dd.ibo.indexType);
          vkCmdDrawIndexedIndirectCount(
              cmd, arg_buf.buffer, dd.multi_draw_indexed_indirect.arg_buf_offset, cnt_buf.buffer,
              dd.multi_draw_indexed_indirect.cnt_buf_offset,
              dd.multi_draw_indexed_indirect.max_count, dd.multi_draw_indexed_indirect.stride);
          break;
        }
        default: {
          UNIMPLEMENTED;
        }
        }
      } else if (type == Cmd_t::EVENT) {
        Deferred_Event de;
        cpu_cmd.read(&de);
        deferred_events.push(de.event);
        // pass
      } else {
        UNIMPLEMENTED;
      }
    }
    vkCmdEndRenderPass(cmd);
    // @TODO: Maybe enable some extension?
    // "vkCmdSetEvent(): It is invalid to issue this call inside an active VkRenderPass"
    ito(deferred_events.size)
        vkCmdSetEvent(cmd, deferred_events[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    cpu_cmd.reset();
  }

  void flush_compute_pass() {
    /* {
       u64      timestamp_results[2];
       VkResult res = vkGetQueryPoolResults(wnd->device, wnd->query_pool, timestamp_begin_id, 2, 16,
                                            timestamp_results, 8, VK_QUERY_RESULT_64_BIT);
       if (res == VK_SUCCESS) {
         u64 diff = timestamp_results[1] - timestamp_results[0];
         last_ms  = diff * wnd->device_properties.limits.timestampPeriod;
       }
     }
     vkResetQueryPool(wnd->device, wnd->query_pool, timestamp_begin_id, 2);
     vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, wnd->query_pool,
                         timestamp_begin_id);*/
    u8  pcs[128];
    u32 pc_dirty_range = 0;
    while (cpu_cmd.has_data()) {
      Cmd_t type = cpu_cmd.read_cmd_type();
      if (type == Cmd_t::TIMESTAMP) {
        Deferred_Timestamp dt;
        cpu_cmd.read(&dt);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dt.pool, dt.timestamp_id);
      } else if (type == Cmd_t::EVENT) {
        Deferred_Event de;
        cpu_cmd.read(&de);
        vkCmdSetEvent(cmd, de.event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      } else if (type == Cmd_t::PUSH_CONSTANTS) {
        Deferred_Push_Constants dpc;
        cpu_cmd.read(&dpc);
        memcpy(pcs + dpc.offset, cpu_cmd.read(dpc.size), dpc.size);
        pc_dirty_range = dpc.offset + dpc.size;
      } else if (type == Cmd_t::BUFFER_FILL) {
        Deferred_Buffer_Fill df;
        cpu_cmd.read(&df);
        Buffer &buf = wnd->buffers[df.buf_id];
        vkCmdFillBuffer(cmd, buf.buffer, df.offset, df.size, df.val);
      } else if (type == Cmd_t::COPY_BUFFER_IMAGE) {
        Deferred_Copy_Buffer_Image dci;
        cpu_cmd.read(&dci);
        _copy_buffer_image(dci.buf_id, dci.offset, dci.img_id, dci.dst_info);
      } else if (type == Cmd_t::COPY_IMAGE_BUFFER) {
        Deferred_Copy_Buffer_Image dci;
        cpu_cmd.read(&dci);
        _copy_image_buffer(dci.buf_id, dci.offset, dci.img_id, dci.dst_info);
      } else if (type == Cmd_t::COPY_BUFFER) {
        Deferred_Copy_Buffer dci;
        cpu_cmd.read(&dci);
        _copy_buffer(dci.src_buf_id, dci.src_offset, dci.dst_buf_id, dci.dst_offset, dci.size);
      } else if (type == Cmd_t::CLEAR_IMAGE) {
        Deferred_Clear_Image cl;
        cpu_cmd.read(&cl);
        Image &img = wnd->images[cl.img_id];
        /* img.barrier(cmd, VK_ACCESS_MEMORY_WRITE_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);*/

        vkCmdClearColorImage(cmd, img.image, img.layout, &cl.cv, 1, &cl.r);
      } else if (type == Cmd_t::DISPATCH) {
        Deferred_Dispatch dd;
        cpu_cmd.read(&dd);
        Compute_Pipeline_Wrapper &gw = wnd->compute_pipelines[dd.pso];
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gw.pipeline);
        if (dd.sets.size > 0)
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gw.pipeline_layout, 0,
                                  dd.sets.size, &dd.sets[0], 0, NULL);

        if (pc_dirty_range != 0) {
          vkCmdPushConstants(cmd, gw.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             pc_dirty_range, pcs);
          pc_dirty_range = 0;
        }
        vkCmdDispatch(cmd, dd.dispatch.x, dd.dispatch.y, dd.dispatch.z);
      } else if (type == Cmd_t::BARRIER) {
        Deferred_Barrier db;
        cpu_cmd.read(&db);
        if (db.image) {
          Image &img = wnd->images[db.res_id];
          img.barrier(cmd, db.new_access_flags, db.new_layout);
        } else {
          Buffer &buffer = wnd->buffers[db.res_id];
          buffer.barrier(cmd, db.new_access_flags);
        }
      } else {
        UNIMPLEMENTED;
      }
    }
    // vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, wnd->query_pool,
    // timestamp_end_id);
    cpu_cmd.reset();
  }

  Graphics_Pipeline_Wrapper &get_or_bake_graphics_pipeline() {
    if (!wnd->pipeline_cache.contains(graphics_state)) {
      Graphics_Pipeline_Wrapper gw;
      ASSERT_DEBUG(!graphics_state.ps.is_null());
      ASSERT_DEBUG(!graphics_state.vs.is_null());
      Shader_Info &ps = wnd->shaders[graphics_state.ps];
      Shader_Info &vs = wnd->shaders[graphics_state.vs];
      ASSERT_DEBUG(!cur_pass.is_null());
      Render_Pass &pass = wnd->render_passes[cur_pass];
      gw.init(wnd->device, pass, vs, ps, graphics_state);
      ID pipe_id = wnd->pipelines.push(gw);
      wnd->pipeline_cache.insert(graphics_state, pipe_id);
    }
    ID pipe_id = wnd->pipeline_cache.get(graphics_state);
    ASSERT_DEBUG(!pipe_id.is_null());
    return wnd->pipelines[pipe_id];
  }

  Compute_Pipeline_Wrapper &get_or_bake_compute_pipeline() {
    if (!wnd->compute_pipeline_cache.contains(cs)) {
      Compute_Pipeline_Wrapper gw;
      ASSERT_DEBUG(!cs.is_null());
      Shader_Info &cs_info = wnd->shaders[cs];
      gw.init(wnd->device, cs_info);
      ID pipe_id = wnd->compute_pipelines.push(gw);
      wnd->compute_pipeline_cache.insert(cs, pipe_id);
    }
    ID pipe_id = wnd->compute_pipeline_cache.get(cs);
    ASSERT_DEBUG(!pipe_id.is_null());
    return wnd->compute_pipelines[pipe_id];
  }

  void record_barriers() {
    // Semi-Automatic barriers for compute shader
    if (type == rd::Pass_t::COMPUTE) return;

    deferred_bindings.iter_pairs([&](Resource_Path const &path, Resource_Binding const &rb) {
      if (rb.type == Binding_t::UNIFORM_BUFFER) {
        Buffer &buffer = wnd->buffers[rb.uniform_buffer.buf_id];
        if (type == rd::Pass_t::COMPUTE) {
          push_buffer_barrier(buffer.id, VK_ACCESS_SHADER_READ_BIT);
        } else {
          buffer.barrier(cmd, VK_ACCESS_SHADER_READ_BIT);
        }
      } else if (rb.type == Binding_t::STORAGE_BUFFER) {
        Buffer &buffer = wnd->buffers[rb.uniform_buffer.buf_id];
        if (type == rd::Pass_t::COMPUTE) {
          push_buffer_barrier(buffer.id, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        } else {
          buffer.barrier(cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }

      } else if (rb.type == Binding_t::IMAGE) {
        VkDescriptorImageInfo binfo;
        MEMZERO(binfo);
        Image &img = wnd->images[rb.image.image_id];
        if (type == rd::Pass_t::COMPUTE) {
          push_image_barrier(img.id, VK_ACCESS_SHADER_READ_BIT,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        } else {
          img.barrier(cmd, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

      } else if (rb.type == Binding_t::STORAGE_IMAGE) {
        VkDescriptorImageInfo binfo;
        MEMZERO(binfo);
        Image &img = wnd->images[rb.image.image_id];

        if (type == rd::Pass_t::COMPUTE) {
          push_image_barrier(img.id, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_GENERAL);
        } else {
          img.barrier(cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_IMAGE_LAYOUT_GENERAL);
        }

      } else if (rb.type == Binding_t::SAMPLER) {

      } else {
        UNIMPLEMENTED;
      }
    });
  }

  void update_descriptor_set(u32 index, Hash_Set<Pair<u32, u32>> *bindings, VkDescriptorSet set) {
    deferred_bindings.iter_pairs([&](Resource_Path const &path, Resource_Binding const &rb) {
      if (bindings->contains({path.set, path.binding}) == false) return;
      if (rb.set != index) return;
      if (rb.type == Binding_t::UNIFORM_BUFFER) {
        Buffer &               buffer = wnd->buffers[rb.uniform_buffer.buf_id];
        VkDescriptorBufferInfo binfo;
        MEMZERO(binfo);
        binfo.buffer = buffer.buffer;
        binfo.offset = rb.uniform_buffer.offset;
        binfo.range  = rb.uniform_buffer.size;
        VkWriteDescriptorSet wset;
        MEMZERO(wset);
        wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wset.descriptorCount = 1;
        wset.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        wset.dstArrayElement = rb.element;
        wset.dstBinding      = rb.binding;
        wset.dstSet          = set;
        wset.pBufferInfo     = &binfo;
        vkUpdateDescriptorSets(wnd->device, 1, &wset, 0, NULL);
      } else if (rb.type == Binding_t::STORAGE_BUFFER) {
        Buffer &buffer = wnd->buffers[rb.uniform_buffer.buf_id];

        VkDescriptorBufferInfo binfo;
        MEMZERO(binfo);
        binfo.buffer = buffer.buffer;
        binfo.offset = rb.storage_buffer.offset;
        binfo.range  = rb.storage_buffer.size;
        VkWriteDescriptorSet wset;
        MEMZERO(wset);
        wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wset.descriptorCount = 1;
        wset.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wset.dstArrayElement = rb.element;
        wset.dstBinding      = rb.binding;
        wset.dstSet          = set;
        wset.pBufferInfo     = &binfo;
        vkUpdateDescriptorSets(wnd->device, 1, &wset, 0, NULL);
      } else if (rb.type == Binding_t::IMAGE) {
        VkDescriptorImageInfo binfo;
        MEMZERO(binfo);
        Image &img     = wnd->images[rb.image.image_id];
        ID     view_id = wnd->create_image_view(img.id, rb.image.sr.baseMipLevel,
                                            rb.image.sr.levelCount, rb.image.sr.baseArrayLayer,
                                            rb.image.sr.layerCount, rb.image.format)
                         .id;
        ImageView &view   = wnd->image_views[view_id];
        binfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        binfo.imageView   = view.view;
        binfo.sampler     = VK_NULL_HANDLE;
        VkWriteDescriptorSet wset;
        MEMZERO(wset);
        wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wset.descriptorCount = 1;
        wset.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        wset.dstArrayElement = rb.element;
        wset.dstBinding      = rb.binding;
        wset.dstSet          = set;
        wset.pImageInfo      = &binfo;
        vkUpdateDescriptorSets(wnd->device, 1, &wset, 0, NULL);
      } else if (rb.type == Binding_t::STORAGE_IMAGE) {
        VkDescriptorImageInfo binfo;
        MEMZERO(binfo);
        Image &img     = wnd->images[rb.image.image_id];
        ID     view_id = wnd->create_image_view(img.id, rb.image.sr.baseMipLevel,
                                            rb.image.sr.levelCount, rb.image.sr.baseArrayLayer,
                                            rb.image.sr.layerCount, rb.image.format)
                         .id;
        ImageView &view   = wnd->image_views[view_id];
        binfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        binfo.imageView   = view.view;
        binfo.sampler     = VK_NULL_HANDLE;
        VkWriteDescriptorSet wset;
        MEMZERO(wset);
        wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wset.descriptorCount = 1;
        wset.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wset.dstArrayElement = rb.element;
        wset.dstBinding      = rb.binding;
        wset.dstSet          = set;
        wset.pImageInfo      = &binfo;
        vkUpdateDescriptorSets(wnd->device, 1, &wset, 0, NULL);
      } else if (rb.type == Binding_t::SAMPLER) {
        VkDescriptorImageInfo binfo;
        MEMZERO(binfo);
        Sampler &sampler  = wnd->samplers[rb.sampler.sampler_id];
        binfo.imageLayout = VK_IMAGE_LAYOUT_MAX_ENUM;
        binfo.imageView   = VK_NULL_HANDLE;
        binfo.sampler     = sampler.sampler;
        VkWriteDescriptorSet wset;
        MEMZERO(wset);
        wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wset.descriptorCount = 1;
        wset.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
        wset.dstArrayElement = rb.element;
        wset.dstBinding      = rb.binding;
        wset.dstSet          = set;
        wset.pImageInfo      = &binfo;
        vkUpdateDescriptorSets(wnd->device, 1, &wset, 0, NULL);
      } else {
        UNIMPLEMENTED;
      }
    });
  }

  public:
  // u64  get_dt() { return last_ms; }
  void reset() {
    vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    this->cur_pass = {0};
    clear_state();
  }

  void init(rd::Pass_t type, Window *wnd, ID pass_id) {
    this->type = type;
    // timestamp_begin_id = wnd->allocate_timestamp_id();
    // timestamp_end_id   = wnd->allocate_timestamp_id();
    set_dirty_flags = 0;
    memset(set_cache, 0, sizeof(set_cache));
    deferred_bindings.init();
    cpu_cmd.init();
    this->wnd = wnd;
    stack.init();
    graphics_state.reset();

    cmd_id = wnd->create_command_buffer();
    cmd    = wnd->cmd_buffers[cmd_id.id].cmd;

    this->cur_pass = pass_id;
    clear_state();
    VkCommandBufferBeginInfo begin_info;
    MEMZERO(begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_ASSERT_OK(vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
    VK_ASSERT_OK(vkBeginCommandBuffer(cmd, &begin_info));
  }

  ID submit(VkFence finish_fence, VkSemaphore *wait_sem) {
    if (type == rd::Pass_t::COMPUTE) {
      flush_compute_pass();
    } else if (type == rd::Pass_t::RENDER) {
      if (cur_pass.is_null() == false) flush_render_pass();
    } else {
      UNIMPLEMENTED;
    }
    vkEndCommandBuffer(cmd);
    VkPipelineStageFlags stage_flags[]{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    VkSubmitInfo         submit_info;
    MEMZERO(submit_info);
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    if (wait_sem != NULL) {
      submit_info.waitSemaphoreCount = 1;
      submit_info.pWaitSemaphores    = wait_sem;
    }
    submit_info.pWaitDstStageMask    = stage_flags;
    submit_info.commandBufferCount   = 1;
    submit_info.pCommandBuffers      = &cmd;
    submit_info.signalSemaphoreCount = 1;

    ID finish_sem;
    finish_sem                    = wnd->create_semaphore().id;
    VkSemaphore raw_sem           = wnd->semaphores[finish_sem].sem;
    submit_info.pSignalSemaphores = &raw_sem;
    vkQueueSubmit(wnd->queue, 1, &submit_info, finish_fence);
    return finish_sem;
  }
  void release() {
    cpu_cmd.release();
    deferred_bindings.release();
    stack.release();
    wnd->release_resource(cmd_id);
    delete this;
  }
  // CTX
  void set_viewport(float x, float y, float width, float height, float mindepth,
                    float maxdepth) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    VkViewport viewports[1];
    viewports[0].x        = x;
    viewports[0].y        = y;
    viewports[0].width    = width;
    viewports[0].height   = height;
    viewports[0].minDepth = mindepth;
    viewports[0].maxDepth = maxdepth;
    Deferred_Rect dr;
    MEMZERO(dr);
    dr.is_viewport = true;
    dr.viewport    = viewports[0];
    cpu_cmd.write(Cmd_t::RENDER_RECT, &dr);
  }
  void set_scissor(u32 x, u32 y, u32 width, u32 height) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    VkRect2D scissors[1];
    scissors[0].offset.x      = x;
    scissors[0].offset.y      = y;
    scissors[0].extent.width  = width;
    scissors[0].extent.height = height;
    Deferred_Rect dr;
    MEMZERO(dr);
    dr.is_viewport = false;
    dr.scissor     = scissors[0];
    cpu_cmd.write(Cmd_t::RENDER_RECT, &dr);
  }

  void clear_state() override {
    set_dirty_flags = 0;
    memset(set_cache, 0, sizeof(set_cache));
    graphics_state.reset();
    deferred_bindings.reset();
    current_draw.reset();
    current_dispatch.reset();
  }

  void clear_bindings() override {
    set_dirty_flags = 0;
    memset(set_cache, 0, sizeof(set_cache));
    deferred_bindings.reset();
  }

  void        push_state() override { stack.push(graphics_state); }
  void        pop_state() override { graphics_state = stack.pop(); }
  Resource_ID insert_event() override {
    Resource_ID    event = wnd->create_event();
    Deferred_Event de;
    MEMZERO(de);
    de.event = wnd->events[event.id].event;
    cpu_cmd.write(Cmd_t::EVENT, &de);
    return event;
  }
  bool get_event_state(Resource_ID fence_id) override {
    Event &  event    = wnd->events[fence_id.id];
    VkResult wait_res = vkGetEventStatus(wnd->device, event.event);
    ASSERT_DEBUG(wait_res == VK_EVENT_SET || wait_res == VK_EVENT_RESET);
    return wait_res == VK_EVENT_SET;
  }
  void IA_set_topology(rd::Primitive topology) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    graphics_state.topology = to_vk(topology);
  }
  void IA_set_index_buffer(Resource_ID res_id, u32 offset, rd::Index_t format) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    if (res_id.is_null()) {
      UNIMPLEMENTED;
    }
    ASSERT_DEBUG(res_id.type == (i32)Resource_Type::BUFFER);
    Buffer &buf = wnd->buffers[res_id.id];
    buf.barrier(cmd, VK_ACCESS_INDEX_READ_BIT);
    VkIndexType type;
    switch (format) {
    case rd::Index_t::UINT32: type = VK_INDEX_TYPE_UINT32; break;
    case rd::Index_t::UINT16: type = VK_INDEX_TYPE_UINT16; break;
    default: TRAP;
    }
    current_draw.ibo.buffer    = buf.buffer;
    current_draw.ibo.offset    = (VkDeviceSize)offset;
    current_draw.ibo.indexType = type;
  }
  void IA_set_vertex_buffer(u32 index, Resource_ID res_id, size_t offset, size_t stride,
                            rd::Input_Rate rate) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    if (res_id.is_null()) {
      UNIMPLEMENTED;
    }
    ASSERT_DEBUG(res_id.type == (i32)Resource_Type::BUFFER);
    Buffer &buf = wnd->buffers[res_id.id];
    buf.barrier(cmd, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
    VkDeviceSize doffset                   = (VkDeviceSize)offset;
    current_draw.vbos.size                 = MAX(index + 1, current_draw.vbos.size);
    current_draw.vbos[index].buffer        = buf.buffer;
    current_draw.vbos[index].offset        = (VkDeviceSize)offset;
    graphics_state.bindings[index].binding = index;
    if (rate == rd::Input_Rate::VERTEX)
      graphics_state.bindings[index].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    else if (rate == rd::Input_Rate::INSTANCE)
      graphics_state.bindings[index].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    graphics_state.bindings[index].stride = stride;
  }
  void IA_set_attribute(rd::Attribute_Info const &info) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    graphics_state.attributes[info.location].binding  = info.binding;
    graphics_state.attributes[info.location].format   = to_vk(info.format);
    graphics_state.attributes[info.location].location = info.location;
    graphics_state.attributes[info.location].offset   = info.offset;
  }
  void VS_set_shader(Resource_ID id) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    if (graphics_state.vs != id.id) {
      memset(set_cache, 0, sizeof(set_cache));
    }
    graphics_state.vs = id.id;
  }
  void PS_set_shader(Resource_ID id) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    if (graphics_state.ps != id.id) {
      memset(set_cache, 0, sizeof(set_cache));
    }
    graphics_state.ps = id.id;
  }
  void CS_set_shader(Resource_ID id) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    if (cs != id.id) {
      memset(set_cache, 0, sizeof(set_cache));
    }
    cs = id.id;
  }
  void RS_set_state(rd::RS_State const &rs_state) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    graphics_state.rs_state = rs_state;
  }
  void DS_set_state(rd::DS_State const &ds_state) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    graphics_state.ds_state = ds_state;
  }
  void MS_set_state(rd::MS_State const &ms_state) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    graphics_state.ms_state = ms_state;
  }
  void RS_set_line_width(float width) override { current_draw.line_width = width; }
  void RS_set_depth_bias(float b) override { current_draw.depth_bias = b; }
  void OM_set_blend_state(u32 rt_index, rd::Blend_State const &bl) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    graphics_state.num_rts = MAX(graphics_state.num_rts, rt_index + 1);
    VkPipelineColorBlendAttachmentState bs;
    MEMZERO(bs);
    bs.blendEnable         = to_vk(bl.enabled);
    bs.srcColorBlendFactor = to_vk(bl.src_color);
    bs.dstColorBlendFactor = to_vk(bl.dst_color);
    bs.colorBlendOp        = to_vk(bl.color_blend_op);
    bs.srcAlphaBlendFactor = to_vk(bl.src_alpha);
    bs.dstAlphaBlendFactor = to_vk(bl.dst_alpha);
    bs.alphaBlendOp        = to_vk(bl.alpha_blend_op);

    if (bl.color_write_mask & (u32)rd::Color_Component_Bit::R_BIT)
      bs.colorWriteMask |= VK_COLOR_COMPONENT_R_BIT;
    if (bl.color_write_mask & (u32)rd::Color_Component_Bit::G_BIT)
      bs.colorWriteMask |= VK_COLOR_COMPONENT_G_BIT;
    if (bl.color_write_mask & (u32)rd::Color_Component_Bit::B_BIT)
      bs.colorWriteMask |= VK_COLOR_COMPONENT_B_BIT;
    if (bl.color_write_mask & (u32)rd::Color_Component_Bit::A_BIT)
      bs.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;

    graphics_state.blend_states[rt_index] = bs;
  }

  void insert_binding(Resource_Binding rb) {
    // Check for redundant bindings
    if (deferred_bindings.contains({rb.set, rb.binding, rb.element})) {
      if (deferred_bindings.get({rb.set, rb.binding, rb.element}) == rb) return;
    }
    set_dirty_flags |= (1 << rb.set);
    deferred_bindings.insert({rb.set, rb.binding, rb.element}, rb);
  }

  void bind_uniform_buffer(u32 set, u32 binding, Resource_ID buf_id, size_t offset,
                           size_t size) override {
    Resource_Binding rb;
    MEMZERO(rb);
    rb.type                  = Binding_t::UNIFORM_BUFFER;
    rb.set                   = set;
    rb.binding               = binding;
    rb.element               = 0;
    rb.uniform_buffer.buf_id = buf_id.id;
    rb.uniform_buffer.offset = offset;
    if (size == 0) size = VK_WHOLE_SIZE;
    rb.uniform_buffer.size = size;
    insert_binding(rb);
  }
  void bind_storage_buffer(u32 set, u32 binding, Resource_ID buf_id, size_t offset,
                           size_t size) override {
    Resource_Binding rb;
    MEMZERO(rb);
    rb.type                  = Binding_t::STORAGE_BUFFER;
    rb.set                   = set;
    rb.binding               = binding;
    rb.element               = 0;
    rb.storage_buffer.buf_id = buf_id.id;
    rb.storage_buffer.offset = offset;
    if (size == 0) size = VK_WHOLE_SIZE;
    rb.storage_buffer.size = size;
    insert_binding(rb);
  }
  Resource_ID insert_timestamp() override {
    Deferred_Timestamp dt;
    MEMZERO(dt);
    dt.pool         = wnd->query_pool;
    dt.timestamp_id = wnd->allocate_timestamp_id();
    vkResetQueryPool(wnd->device, dt.pool, dt.timestamp_id, 1);
    cpu_cmd.write(Cmd_t::TIMESTAMP, &dt);
    return {dt.timestamp_id, (u32)Resource_Type::TIMESTAMP};
  }
  void bind_image(u32 set, u32 binding, u32 index, Resource_ID image_id,
                  rd::Image_Subresource const &range, rd::Format format) override {
    Image &          image = wnd->images[image_id.id];
    Resource_Binding rb;
    MEMZERO(rb);
    rb.type                    = Binding_t::IMAGE;
    rb.set                     = set;
    rb.binding                 = binding;
    rb.element                 = index;
    rb.image.image_id          = image_id.id;
    rb.image.sr.aspectMask     = image.aspect;
    rb.image.sr.baseArrayLayer = range.layer;
    rb.image.sr.layerCount     = range.num_layers;
    rb.image.sr.baseMipLevel   = range.level;
    rb.image.sr.levelCount     = range.num_levels;
    if (format == rd::Format::NATIVE)
      rb.image.format = image.info.format;
    else
      rb.image.format = to_vk(format);
    insert_binding(rb);
  }
  void bind_sampler(u32 set, u32 binding, Resource_ID sampler_id) override {
    Resource_Binding rb;
    MEMZERO(rb);
    rb.type               = Binding_t::SAMPLER;
    rb.set                = set;
    rb.binding            = binding;
    rb.element            = 0;
    rb.sampler.sampler_id = sampler_id.id;
    insert_binding(rb);
  }
  void bind_rw_image(u32 set, u32 binding, u32 index, Resource_ID image_id,
                     rd::Image_Subresource const &range, rd::Format format) override {
    Image &          image = wnd->images[image_id.id];
    Resource_Binding rb;
    MEMZERO(rb);
    rb.type                    = Binding_t::STORAGE_IMAGE;
    rb.set                     = set;
    rb.binding                 = binding;
    rb.element                 = index;
    rb.image.image_id          = image_id.id;
    rb.image.sr.aspectMask     = image.aspect;
    rb.image.sr.baseArrayLayer = range.layer;
    rb.image.sr.layerCount     = range.num_layers;
    rb.image.sr.baseMipLevel   = range.level;
    rb.image.sr.levelCount     = range.num_levels;
    if (format == rd::Format::NATIVE)
      rb.image.format = image.info.format;
    else
      rb.image.format = to_vk(format);
    insert_binding(rb);
  }
  void push_constants(void const *data, size_t offset, size_t size) override {
    Deferred_Push_Constants pc;
    pc.offset = offset;
    pc.size   = size;
    cpu_cmd.write(Cmd_t::PUSH_CONSTANTS, &pc);
    cpu_cmd.write(data, size);
  }
  void bake_descriptor_sets() {
    VkDescriptorSet *         set_dst     = NULL;
    VkDescriptorSetLayout *   set_layouts = NULL;
    u32                       num_sets    = 0;
    Hash_Set<Pair<u32, u32>> *bindings    = NULL;
    if (type == rd::Pass_t::RENDER) {
      Graphics_Pipeline_Wrapper &gw = get_or_bake_graphics_pipeline();
      current_draw.sets.resize(gw.set_layouts.size);
      num_sets    = gw.set_layouts.size;
      set_layouts = &gw.set_layouts[0];
      set_dst     = &current_draw.sets[0];
      bindings    = &gw.bindings;
    } else {
      Compute_Pipeline_Wrapper &gw = get_or_bake_compute_pipeline();
      current_dispatch.sets.resize(gw.set_layouts.size);
      num_sets    = gw.set_layouts.size;
      set_layouts = &gw.set_layouts[0];
      set_dst     = &current_dispatch.sets[0];
      bindings    = &gw.bindings;
    }
    record_barriers();
    ito(num_sets) {
      if (set_layouts[i] == VK_NULL_HANDLE) continue;
      if ((set_dirty_flags & (1 << i)) == 0 && set_cache[i] != VK_NULL_HANDLE) {
        set_dst[i] = set_cache[i];
      } else {
        VkDescriptorSet set = wnd->get_descriptor_pool().allocate(set_layouts[i]);
        update_descriptor_set(i, bindings, set);
        set_dirty_flags &= ~(1 << i);
        set_cache[i] = set;
        set_dst[i]   = set;
      }
    }
  }
  void copy_image_to_buffer(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    if (type == rd::Pass_t::COMPUTE) {
      Deferred_Copy_Buffer_Image dci;
      MEMZERO(dci);
      dci.buf_id   = buf_id;
      dci.dst_info = dst_info;
      dci.offset   = offset;
      dci.img_id   = img_id;
      cpu_cmd.write(Cmd_t::COPY_IMAGE_BUFFER, &dci);
    } else {
      _copy_image_buffer(buf_id, offset, img_id, dst_info);
    }
  }
  void copy_buffer_to_image(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    if (type == rd::Pass_t::COMPUTE) {
      Deferred_Copy_Buffer_Image dci;
      MEMZERO(dci);
      dci.buf_id   = buf_id;
      dci.dst_info = dst_info;
      dci.offset   = offset;
      dci.img_id   = img_id;
      cpu_cmd.write(Cmd_t::COPY_BUFFER_IMAGE, &dci);
    } else {
      _copy_buffer_image(buf_id, offset, img_id, dst_info);
    }
  }
  void copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                   size_t dst_offset, u32 size) override {
    if (type == rd::Pass_t::COMPUTE) {
      Deferred_Copy_Buffer dci;
      MEMZERO(dci);
      dci.src_buf_id = src_buf_id;
      dci.src_offset = src_offset;
      dci.dst_buf_id = dst_buf_id;
      dci.dst_offset = dst_offset;
      dci.size       = size;
      cpu_cmd.write(Cmd_t::COPY_BUFFER, &dci);
    } else {
      _copy_buffer(src_buf_id, src_offset, dst_buf_id, dst_offset, size);
    }
  }
  void draw_indexed(u32 index_count, u32 instance_count, u32 first_index, u32 first_instance,
                    i32 vertex_offset) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    bake_descriptor_sets();
    current_draw.pso                         = get_or_bake_graphics_pipeline().id;
    current_draw.type                        = Deferred_Draw::Type::DRAW_INDEXED;
    current_draw.draw_indexed.index_count    = index_count;
    current_draw.draw_indexed.instance_count = instance_count;
    current_draw.draw_indexed.first_index    = first_index;
    current_draw.draw_indexed.first_instance = first_instance;
    current_draw.draw_indexed.vertex_offset  = vertex_offset;
    cpu_cmd.write(Cmd_t::DRAW, &current_draw);
  }
  void draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    bake_descriptor_sets();
    current_draw.pso                 = get_or_bake_graphics_pipeline().id;
    current_draw.type                = Deferred_Draw::Type::DRAW;
    current_draw.draw.vertex_count   = vertex_count;
    current_draw.draw.instance_count = instance_count;
    current_draw.draw.first_vertex   = first_vertex;
    current_draw.draw.first_instance = first_instance;
    cpu_cmd.write(Cmd_t::DRAW, &current_draw);
  }
  void multi_draw_indexed_indirect(Resource_ID arg_buf_id, u32 arg_buf_offset,
                                   Resource_ID cnt_buf_id, u32 cnt_buf_offset, u32 max_count,
                                   u32 stride) override {
    {
      Buffer &buffer = wnd->buffers[arg_buf_id.id];
      buffer.barrier(cmd, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    }
    {
      Buffer &buffer = wnd->buffers[cnt_buf_id.id];
      buffer.barrier(cmd, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    }
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    bake_descriptor_sets();
    current_draw.pso  = get_or_bake_graphics_pipeline().id;
    current_draw.type = Deferred_Draw::Type::MULTI_DRAW_INDEXED_INDIRECT;
    current_draw.multi_draw_indexed_indirect.arg_buf_id     = arg_buf_id.id;
    current_draw.multi_draw_indexed_indirect.arg_buf_offset = arg_buf_offset;
    current_draw.multi_draw_indexed_indirect.cnt_buf_id     = cnt_buf_id.id;
    current_draw.multi_draw_indexed_indirect.cnt_buf_offset = cnt_buf_offset;
    current_draw.multi_draw_indexed_indirect.max_count      = max_count;
    current_draw.multi_draw_indexed_indirect.stride         = stride;
    cpu_cmd.write(Cmd_t::DRAW, &current_draw);
  }
  virtual void image_barrier(Resource_ID image_id, u32 access_flags,
                             rd::Image_Layout layout) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    push_image_barrier(image_id.id, to_vk_access_flags(access_flags), to_vk(layout));
  }
  virtual void buffer_barrier(Resource_ID buf_id, u32 access_flags) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    push_buffer_barrier(buf_id.id, to_vk_access_flags(access_flags));
  }
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    bake_descriptor_sets();
    current_dispatch.pso        = get_or_bake_compute_pipeline().id;
    current_dispatch.dispatch.x = dim_x;
    current_dispatch.dispatch.y = dim_y;
    current_dispatch.dispatch.z = dim_z;
    cpu_cmd.write(Cmd_t::DISPATCH, &current_dispatch);
  }
  void fill_buffer(Resource_ID id, size_t offset, size_t size, u32 value) override {
    if (type == rd::Pass_t::COMPUTE) {
      Deferred_Buffer_Fill bf;
      MEMZERO(bf);
      bf.buf_id = id.id;
      bf.offset = offset;
      if (size == 0) size = VK_WHOLE_SIZE;
      bf.size = size;
      bf.val  = value;
      cpu_cmd.write(Cmd_t::BUFFER_FILL, &bf);
    } else {
      Buffer &buf = wnd->buffers[id.id];
      vkCmdFillBuffer(cmd, buf.buffer, offset, size, value);
    }
  }
  void clear_image(Resource_ID id, rd::Image_Subresource const &range,
                   rd::Clear_Value const &cv) override {
    if (type == rd::Pass_t::COMPUTE) {
      Deferred_Clear_Image cl;
      MEMZERO(cl);
      Image &           img = wnd->images[id.id];
      VkClearColorValue _cv;
      MEMZERO(_cv);
      memcpy(&_cv, &cv, 16);
      VkImageSubresourceRange r;
      MEMZERO(r);
      r.aspectMask     = img.aspect;
      r.baseArrayLayer = range.layer;
      r.layerCount     = range.num_layers;
      r.baseMipLevel   = range.level;
      r.levelCount     = range.num_levels;
      cl.cv            = _cv;
      cl.r             = r;
      cl.img_id        = id.id;
      cpu_cmd.write(Cmd_t::CLEAR_IMAGE, &cl);
    } else {
      Image &img = wnd->images[id.id];
      img.barrier(cmd, VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      VkClearColorValue _cv;
      MEMZERO(_cv);
      memcpy(&_cv, &cv, 16);
      VkImageSubresourceRange r;
      MEMZERO(r);
      r.aspectMask     = img.aspect;
      r.baseArrayLayer = range.layer;
      r.layerCount     = range.num_layers;
      r.baseMipLevel   = range.level;
      r.levelCount     = range.num_levels;
      vkCmdClearColorImage(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &_cv, 1, &r);
    }
  }
  rd::Image_Info get_image_info(Resource_ID res_id) override { return wnd->get_image_info(res_id); }
};

class VkFactory : public rd::IFactory {
  Window *           wnd;
  Array<Resource_ID> release_queue;

  public:
  ID   last_sem;
  void init(Window *wnd) {
    this->wnd = wnd;
    last_sem  = {0};
    release_queue.init();
  }
  void release() {
    release_queue.release();
    delete this;
  }
  void on_frame_begin() { last_sem = {0}; }
  void on_frame_end() {
    if (last_sem.is_null() == false) {
      wnd->release_resource({last_sem, (u32)Resource_Type::SEMAPHORE});
    }
    last_sem = {0};
    ito(release_queue.size) wnd->release_resource(release_queue[i]);
    release_queue.release();
  }
  bool get_timestamp_state(Resource_ID t0) override {
    u64      timestamp_results[1];
    VkResult res = vkGetQueryPoolResults(wnd->device, wnd->query_pool, t0.id._id, 1, 8,
                                         timestamp_results, 8, VK_QUERY_RESULT_64_BIT);
    return res == VK_SUCCESS;
  }
  double get_timestamp_ms(Resource_ID t0, Resource_ID t1) override {
    u64      timestamp_results[2];
    VkResult res0 = vkGetQueryPoolResults(wnd->device, wnd->query_pool, t0.id._id, 1, 8,
                                          timestamp_results + 0, 8, VK_QUERY_RESULT_64_BIT);
    VkResult res1 = vkGetQueryPoolResults(wnd->device, wnd->query_pool, t1.id._id, 1, 8,
                                          timestamp_results + 1, 8, VK_QUERY_RESULT_64_BIT);
    if (res0 == VK_SUCCESS && res1 == VK_SUCCESS) {
      u64 diff = timestamp_results[1] - timestamp_results[0];
      return (double(diff) * wnd->device_properties.limits.timestampPeriod) * 1.0e-6;
    }
    return 0.0;
  }
  Resource_ID create_image(rd::Image_Create_Info info) override {
    return wnd->create_image(info.width, info.height, info.depth, info.layers, info.levels,
                             to_vk(info.format), info.usage_bits, info.mem_bits);
  }
  Resource_ID create_buffer(rd::Buffer_Create_Info info) override {
    return wnd->create_buffer(info);
  }
  Resource_ID create_shader_raw(rd::Stage_t type, string_ref body,
                                Pair<string_ref, string_ref> *defines,
                                size_t                        num_defines) override {

    return wnd->create_shader_raw(type, body, defines, num_defines);
  }
  void *      map_buffer(Resource_ID res_id) override { return wnd->map_buffer(res_id); }
  void        unmap_buffer(Resource_ID res_id) override { wnd->unmap_buffer(res_id); }
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) override {
    return wnd->create_sampler(info);
  }
  void        release_resource(Resource_ID id) override { release_queue.push(id); }
  Resource_ID get_swapchain_image() override {
    return {wnd->sc_images[wnd->image_index], (u32)Resource_Type::IMAGE};
  }
  rd::Image2D_Info get_swapchain_image_info() override { return wnd->get_swapchain_image_info(); }
  rd::Image_Info get_image_info(Resource_ID res_id) override { return wnd->get_image_info(res_id); }

  rd::Imm_Ctx *start_render_pass(rd::Render_Pass_Create_Info const &info) override {
    ID pass_id = {0};
    if (wnd->pass_cache.contains(info)) {
      pass_id = wnd->pass_cache.get(info);
      wnd->render_passes.add_ref(pass_id);
    } else {
      Render_Pass rp;
      rp.init();
      rp.create_info          = info;
      u32 depth_attachment_id = 0;
      ito(info.rts.size) {
        VkFormat format;
        if (info.rts[i].format == rd::Format::NATIVE)
          format = wnd->images[info.rts[i].image.id].info.format;
        else
          format = to_vk(info.rts[i].format);
        rp.rts_views.push(wnd->create_image_view(info.rts[i].image.id, //
                                                 info.rts[i].level, 1, //
                                                 info.rts[i].layer, 1, //
                                                 format)
                              .id);
      }
      if (info.depth_target.image.is_null() == false) {
        VkFormat format;
        if (info.depth_target.format == rd::Format::NATIVE)
          format = wnd->images[info.depth_target.image.id].info.format;
        else
          format = to_vk(info.depth_target.format);
        rp.depth_target_view = wnd->create_image_view(info.depth_target.image.id, //
                                                      info.depth_target.level, 1, //
                                                      info.depth_target.layer, 1, //
                                                      format)
                                   .id;
      }
      InlineArray<VkAttachmentDescription, 9> attachments;
      InlineArray<VkAttachmentReference, 8>   refs;
      attachments.init();
      refs.init();
      defer({
        attachments.release();
        refs.release();
      });
      u32 width  = info.width;
      u32 height = info.height;
      ito(info.rts.size) {
        VkAttachmentDescription attachment;
        MEMZERO(attachment);
        Image &img = wnd->images[info.rts[i].image.id];
        if (width == 0)
          width = img.info.extent.width;
        else
          ASSERT_ALWAYS(width == img.info.extent.width);
        if (height == 0)
          height = img.info.extent.height;
        else
          ASSERT_ALWAYS(height == img.info.extent.height);

        attachment.format  = img.info.format;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        if (info.rts[i].clear_color.clear)
          attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        else
          attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        rp.clear_values.push(to_vk(info.rts[i].clear_color));
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

        attachments.push(attachment);
      }
      if (info.depth_target.image.is_null() == false) {
        VkAttachmentDescription attachment;
        MEMZERO(attachment);
        Image &img = wnd->images[info.depth_target.image.id];
        if (width == 0)
          width = img.info.extent.width;
        else
          ASSERT_ALWAYS(width == img.info.extent.width);
        if (height == 0)
          height = img.info.extent.height;
        else
          ASSERT_ALWAYS(height == img.info.extent.height);

        attachment.format  = img.info.format;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        if (info.depth_target.clear_depth.clear)
          attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        else
          attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        rp.clear_values.push(to_vk(info.depth_target.clear_depth));
        attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_id       = attachments.size;
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
      if (info.depth_target.image.is_null() == false) {
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

      rp.height = height;
      rp.width  = width;

      VK_ASSERT_OK(vkCreateRenderPass(wnd->device, &cinfo, NULL, &rp.pass));
      {
        InlineArray<VkImageView, 8> views;
        views.init();
        defer(views.release());
        ito(rp.rts_views.size) { views.push(wnd->image_views[rp.rts_views[i]].view); }
        if (info.depth_target.image.is_null() == false) {
          views.push(wnd->image_views[rp.depth_target_view].view);
        }
        VkFramebufferCreateInfo info;
        MEMZERO(info);
        info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.attachmentCount = views.size;
        info.width           = width;
        info.height          = height;
        info.layers          = 1;
        info.pAttachments    = &views[0];
        info.renderPass      = rp.pass;
        VK_ASSERT_OK(vkCreateFramebuffer(wnd->device, &info, NULL, &rp.fb));
      }
      pass_id = wnd->render_passes.push(rp);
      wnd->pass_cache.insert(rp.create_info, pass_id);
    }
    Vk_Ctx *ctx = new Vk_Ctx();
    ctx->init(rd::Pass_t::RENDER, wnd, pass_id);
    return ctx;
  }
  void end_render_pass(rd::Imm_Ctx *_ctx) override {
    Vk_Ctx *    ctx = (Vk_Ctx *)_ctx;
    Resource_ID f   = wnd->create_fence(false);
    ID          finish_sem;
    if (last_sem.is_null() == false) {
      Semaphore s = wnd->semaphores[last_sem];
      finish_sem  = ctx->submit(wnd->fences[f.id].fence, &s.sem);
      wnd->release_resource({last_sem, (u32)Resource_Type::SEMAPHORE});
      last_sem = {0};
    } else {
      finish_sem = ctx->submit(wnd->fences[f.id].fence, NULL);
    }
    last_sem = finish_sem;
    wnd->release_resource(f);
    ctx->release();
  }
  rd::Imm_Ctx *start_compute_pass() override {
    Vk_Ctx *ctx = new Vk_Ctx();
    ctx->init(rd::Pass_t::COMPUTE, wnd, {0});
    return ctx;
  }
  void wait_idle() { vkDeviceWaitIdle(wnd->device); }
  void end_compute_pass(rd::Imm_Ctx *_ctx) override {
    Vk_Ctx *    ctx = (Vk_Ctx *)_ctx;
    Resource_ID f   = wnd->create_fence(false);

    ID finish_sem;
    if (last_sem.is_null() == false) {
      Semaphore s = wnd->semaphores[last_sem];
      finish_sem  = ctx->submit(wnd->fences[f.id].fence, &s.sem);
      wnd->release_resource({last_sem, (u32)Resource_Type::SEMAPHORE});
      last_sem = {0};
    } else {
      finish_sem = ctx->submit(wnd->fences[f.id].fence, NULL);
    }
    last_sem = finish_sem;

    wnd->release_resource(f);
    ctx->release();
  }
};

class VkPass_Mng : public rd::Pass_Mng {
  public:
  Window *             wnd;
  VkFactory *          f;
  rd::IEvent_Consumer *consumer;
  VkPass_Mng() {
    wnd = new Window();
    wnd->init();
    f = new VkFactory();
    f->init(wnd);
    consumer = NULL;
  }
  void release() override {
    consumer->on_release(f);
    f->release();
    wnd->release();
    delete wnd;
    delete this;
  }
  void loop() override {
    consumer->init(this);
    wnd->start_frame();
    consumer->on_init(f);
    wnd->end_frame(NULL);
    while (true) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (consumer != NULL) {
          consumer->consume(&event);
        }
        if (event.type == SDL_QUIT) {
          release();
          exit(0);
        }
        switch (event.type) {
        case SDL_WINDOWEVENT:
          if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
          }
          break;
        }
      }
      wnd->start_frame();
      f->on_frame_begin();
      consumer->on_frame(f);
      VkSemaphore sem = VK_NULL_HANDLE;
      if (f->last_sem.is_null() == false) {
        sem = wnd->semaphores[f->last_sem].sem;
      }
      f->on_frame_end();
      if (sem != VK_NULL_HANDLE)
        wnd->end_frame(&sem);
      else
        wnd->end_frame(NULL);
    }
  }
  void  set_event_consumer(rd::IEvent_Consumer *consumer) override { this->consumer = consumer; }
  void *get_window_handle() { return (void *)wnd->window; }
};
} // namespace
rd::Pass_Mng *rd::Pass_Mng::create(rd::Impl_t) { return new VkPass_Mng; }