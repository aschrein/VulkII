#include "script.hpp"
#include "utils.hpp"

#include <functional>

#define SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS

#include <3rdparty/dxc/dxcapi.h>
#include <mutex>
#include <string>
#include <thread>

#ifdef __linux__
#  define VK_USE_PLATFORM_XCB_KHR
#  include <shaderc/shaderc.h>
#  include <spirv_cross/spirv_cross_c.h>
#  include <vulkan/vulkan.h>
#else
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <atlbase.h>
#  include <shaderc/shaderc.h>
#  include <spirv_cross/spirv_cross_c.h>
#  include <vulkan/vulkan.h>
#  include <vulkan/vulkan_win32.h>
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

// VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT flags,
//                                       VkDebugReportObjectTypeEXT /*objType*/,
//                                       uint64_t /*srcObject*/, size_t /*location*/, int32_t
//                                       msgCode, const char *pLayerPrefix, const char *pMsg, void *
//                                       /*pUserData*/) {
//
//  switch (flags) {
//  case VK_DEBUG_REPORT_INFORMATION_BIT_EXT: fprintf(stdout, "INFORMATION: "); break;
//  case VK_DEBUG_REPORT_WARNING_BIT_EXT: fprintf(stdout, "WARNING: "); break;
//  case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT: fprintf(stdout, "PERFORMANCE WARNING: ");
//  break; case VK_DEBUG_REPORT_ERROR_BIT_EXT: fprintf(stdout, "ERROR: "); break; case
//  VK_DEBUG_REPORT_DEBUG_BIT_EXT: fprintf(stdout, "DEBUG: "); break; default: fprintf(stdout,
//  "unknown flag %i", (int)flags); break;
//  }
//
//  fprintf(stdout, pLayerPrefix);
//  fprintf(stdout, " ");
//  fprintf(stdout, pMsg);
//
//  return false;
//}

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
  void set_index(u32 index) {
    id._id            = index + 1;
    frames_referenced = 0;
  }
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
  SET,
  GRAPHICS_PSO,
  COMPUTE_PSO,
  SIGNATURE,
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
  std::mutex *          mapped_mutex     = NULL;
  VkDeviceMemory        mem              = VK_NULL_HANDLE;
  VkMemoryPropertyFlags prop_flags       = 0;
  u32                   size             = 0;
  u32                   cursor           = 0; // points to the next free 4kb byte block
  u32                   memory_type_bits = 0;

  void dump() {
    fprintf(stdout, "Mem_Chunk {\n");
    fprintf(stdout, "  ref_cnt: %i\n", ref_cnt);
    fprintf(stdout, "  size   : %i\n", size);
    fprintf(stdout, "  cursor : %i\n", cursor);
    fprintf(stdout, "}\n");
  }
  void lock() { mapped_mutex->lock(); }
  void unlock() { mapped_mutex->unlock(); }
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
    mapped_mutex           = new std::mutex;
  }
  void release(VkDevice device) {
    delete mapped_mutex;
    vkFreeMemory(device, mem, NULL);
    MEMZERO(*this);
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

struct DescriptorSet : public Slot {
  VkDescriptorSet set;
  explicit DescriptorSet(VkDescriptorSet set) : set(set) { frames_referenced = 0; }
};

struct Buffer : public Slot {
  string             name;
  ID                 mem_chunk_id;
  u32                mem_offset;
  VkBuffer           buffer;
  VkBufferCreateInfo create_info;

  InlineArray<ID, 8> views;

  void init() {
    memset(this, 0, sizeof(*this));
    views.init();
  }
  void release() {
    views.release();
    name.release();
  }
  Resource_ID get_resource_id() const { return {{id, (u32)Resource_Type::BUFFER}}; }
};

struct BufferLaoutTracker {
  VkAccessFlags access_flags = 0;
  void          barrier(VkCommandBuffer cmd, Buffer *buffer, VkAccessFlags new_access_flags) {
    VkBufferMemoryBarrier bar;
    MEMZERO(bar);
    bar.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bar.srcAccessMask       = access_flags;
    bar.dstAccessMask       = new_access_flags;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer              = buffer->buffer;
    bar.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 1, &bar, 0, NULL);
    access_flags = new_access_flags;
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

// VkAccessFlags to_vk_access_flags(u32 flags) {
//  u32 out = 0;
//  // clang-format off
//  if (flags & rd::Access_Bits::UNIFORM_READ                     ) out |=
//  VK_ACCESS_UNIFORM_READ_BIT                   ; if (flags & rd::Access_Bits::SHADER_READ ) out |=
//  VK_ACCESS_SHADER_READ_BIT                    ; if (flags & rd::Access_Bits::SHADER_WRITE ) out
//  |= VK_ACCESS_SHADER_WRITE_BIT                   ; if (flags & rd::Access_Bits::TRANSFER_READ )
//  out |= VK_ACCESS_TRANSFER_READ_BIT                  ; if (flags &
//  rd::Access_Bits::TRANSFER_WRITE                   ) out |= VK_ACCESS_TRANSFER_WRITE_BIT ; if
//  (flags & rd::Access_Bits::HOST_READ                        ) out |= VK_ACCESS_HOST_READ_BIT ; if
//  (flags & rd::Access_Bits::HOST_WRITE                       ) out |= VK_ACCESS_HOST_WRITE_BIT ;
//  if (flags & rd::Access_Bits::MEMORY_READ                      ) out |= VK_ACCESS_MEMORY_READ_BIT
//  ; if (flags & rd::Access_Bits::MEMORY_WRITE                     ) out |=
//  VK_ACCESS_MEMORY_WRITE_BIT                   ;
//  // clang-format on
//  return out;
//}

VkAccessFlags to_vk(rd::Buffer_Access access) {
  switch (access) {
  case rd::Buffer_Access::GENERIC: return 0;
  case rd::Buffer_Access::HOST_READ: return VK_ACCESS_HOST_READ_BIT;
  case rd::Buffer_Access::HOST_READ_WRITE:
    return VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
  case rd::Buffer_Access::INDEX_BUFFER: return VK_ACCESS_INDEX_READ_BIT;
  case rd::Buffer_Access::TRANSFER_DST: return VK_ACCESS_MEMORY_WRITE_BIT;
  case rd::Buffer_Access::TRANSFER_SRC: return VK_ACCESS_MEMORY_READ_BIT;
  case rd::Buffer_Access::UAV: return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  case rd::Buffer_Access::UNIFORM: return VK_ACCESS_UNIFORM_READ_BIT;
  case rd::Buffer_Access::VERTEX_BUFFER: return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
  default: TRAP;
  }
}

void to_vk(rd::Image_Access access, VkAccessFlags &bits, VkImageLayout &layout) {
  switch (access) {
  case rd::Image_Access::COLOR_TARGET: {
    bits   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    return;
  }
  case rd::Image_Access::DEPTH_TARGET: {
    bits   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    return;
  }
  case rd::Image_Access::GENERIC: {
    bits   = 0;
    layout = VK_IMAGE_LAYOUT_GENERAL;
    return;
  }
  case rd::Image_Access::SAMPLED: {
    bits   = VK_ACCESS_SHADER_READ_BIT;
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return;
  }
  case rd::Image_Access::TRANSFER_DST: {
    bits   = VK_ACCESS_MEMORY_WRITE_BIT;
    layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    return;
  }
  case rd::Image_Access::TRANSFER_SRC: {
    bits   = VK_ACCESS_MEMORY_READ_BIT;
    layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    return;
  }
  case rd::Image_Access::UAV: {
    bits   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    layout = VK_IMAGE_LAYOUT_GENERAL;
    return;
  }
  default: TRAP;
  }
}

VkDescriptorType to_vk(rd::Binding_t type) {
  // clang-format off
  switch (type) {
  case rd::Binding_t::SAMPLER             : return VK_DESCRIPTOR_TYPE_SAMPLER;
  case rd::Binding_t::READ_ONLY_BUFFER    : return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  case rd::Binding_t::TEXTURE             : return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  case rd::Binding_t::UAV_BUFFER          : return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  case rd::Binding_t::UAV_TEXTURE         : return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  case rd::Binding_t::UNIFORM_BUFFER      : return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

VkImageLayout to_vk(rd::Image_Access layout) {
  // clang-format off
  switch (layout) {
  case rd::Image_Access::GENERIC                       : return VK_IMAGE_LAYOUT_GENERAL;
  case rd::Image_Access::COLOR_TARGET                  : return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  case rd::Image_Access::DEPTH_TARGET                  : return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  case rd::Image_Access::UAV                           : return VK_IMAGE_LAYOUT_GENERAL;
  case rd::Image_Access::SAMPLED                       : return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  case rd::Image_Access::TRANSFER_DST                  : return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  case rd::Image_Access::TRANSFER_SRC                  : return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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

  case rd::Format::BGRA8_SRGBA     : return VK_FORMAT_B8G8R8A8_SRGB       ;

  //case rd::Format::RGB8_UNORM      : return VK_FORMAT_R8G8B8_UNORM        ;
  //case rd::Format::RGB8_SNORM      : return VK_FORMAT_R8G8B8_SNORM        ;
  //case rd::Format::RGB8_SRGBA      : return VK_FORMAT_R8G8B8_SRGB         ;
  //case rd::Format::RGB8_UINT       : return VK_FORMAT_R8G8B8_UINT         ;

  case rd::Format::RGBA32_FLOAT    : return VK_FORMAT_R32G32B32A32_SFLOAT ;
  case rd::Format::RGB32_FLOAT     : return VK_FORMAT_R32G32B32_SFLOAT    ;
  case rd::Format::RG32_FLOAT      : return VK_FORMAT_R32G32_SFLOAT       ;
  case rd::Format::R32_FLOAT       : return VK_FORMAT_R32_SFLOAT          ;
  case rd::Format::R16_FLOAT       : return VK_FORMAT_R16_SFLOAT          ;
  case rd::Format::R16_UNORM       : return VK_FORMAT_R16_UNORM;
  case rd::Format::R8_UNORM        : return VK_FORMAT_R8_UNORM;
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
  case VK_FORMAT_B8G8R8A8_SRGB       : return rd::Format::BGRA8_SRGBA     ;
  case VK_FORMAT_B8G8R8_UNORM        : return rd::Format::BGR8_UNORM      ;

  case VK_FORMAT_R8G8B8A8_UNORM      : return rd::Format::RGBA8_UNORM     ;
  case VK_FORMAT_R8G8B8A8_SNORM      : return rd::Format::RGBA8_SNORM     ;
  case VK_FORMAT_R8G8B8A8_SRGB       : return rd::Format::RGBA8_SRGBA     ;
  case VK_FORMAT_R8G8B8A8_UINT       : return rd::Format::RGBA8_UINT      ;

  //case VK_FORMAT_R8G8B8_UNORM        : return rd::Format::RGB8_UNORM      ;
  //case VK_FORMAT_R8G8B8_SNORM        : return rd::Format::RGB8_SNORM      ;
  //case VK_FORMAT_R8G8B8_SRGB         : return rd::Format::RGB8_SRGBA      ;
  //case VK_FORMAT_R8G8B8_UINT         : return rd::Format::RGB8_UINT       ;

  case VK_FORMAT_R32G32B32A32_SFLOAT : return rd::Format::RGBA32_FLOAT    ;
  case VK_FORMAT_R32G32B32_SFLOAT    : return rd::Format::RGB32_FLOAT     ;
  case VK_FORMAT_R32G32_SFLOAT       : return rd::Format::RG32_FLOAT      ;
  case VK_FORMAT_R32_SFLOAT          : return rd::Format::R32_FLOAT       ;
  case VK_FORMAT_D32_SFLOAT          : return rd::Format::D32_FLOAT       ;
  case VK_FORMAT_R32_UINT            : return rd::Format::R32_UINT        ;
  case VK_FORMAT_R16_UINT            : return rd::Format::R16_UINT        ;
  case VK_FORMAT_R8_UNORM            : return rd::Format::R8_UNORM        ;
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
  Resource_ID get_resource_id() const { return {{id, (u32)Resource_Type::IMAGE}}; }
};

struct ImageLayoutTracker {
  VkImageLayout layout       = VK_IMAGE_LAYOUT_GENERAL;
  VkAccessFlags access_flags = 0;
  void          barrier(VkCommandBuffer cmd, Image *image, VkAccessFlags new_access_flags,
                        VkImageLayout new_layout) {
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
    bar.image                           = image->image;
    bar.subresourceRange.aspectMask     = image->aspect;
    bar.subresourceRange.baseArrayLayer = 0;
    bar.subresourceRange.baseMipLevel   = 0;
    bar.subresourceRange.layerCount     = image->info.arrayLayers;
    bar.subresourceRange.levelCount     = image->info.mipLevels;
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

static Array<u32> compile_hlsl(VkDevice device, string_ref text, shaderc_shader_kind kind,
                               Pair<string_ref, string_ref> *defines, size_t num_defines) {
#ifdef WIN32
  static HMODULE dxcompilerDLL = LoadLibraryA("dxcompiler.dll");
  ASSERT_ALWAYS(dxcompilerDLL);
  static DxcCreateInstanceProc DxcCreateInstance =
      (DxcCreateInstanceProc)GetProcAddress(dxcompilerDLL, "DxcCreateInstance");
#else
  static void *dxcompilerDLL = dlopen("libdxcompiler.so", RTLD_NOW);
  ASSERT_ALWAYS(dxcompilerDLL);
  static DxcCreateInstanceProc DxcCreateInstance =
      (DxcCreateInstanceProc)dlsym(dxcompilerDLL, "DxcCreateInstance");
  ASSERT_ALWAYS(DxcCreateInstance);
#endif

  static CComPtr<IDxcLibrary>  library;
  static CComPtr<IDxcCompiler> compiler;
  static int                   init = [&] {
    ASSERT_ALWAYS(S_OK == DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));
    ASSERT_ALWAYS(S_OK == DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library)));
    return 0;
  }();
  CComPtr<IDxcBlobEncoding> blob;
  ASSERT_ALWAYS(S_OK ==
                library->CreateBlobWithEncodingFromPinned(text.ptr, (uint32_t)text.len, 0, &blob));
  LPCWSTR profile = L"ps_6_2";
  if (kind == shaderc_vertex_shader)
    profile = L"vs_6_2";
  else if (kind == shaderc_fragment_shader)
    profile = L"ps_6_2";
  else if (kind == shaderc_compute_shader)
    profile = L"cs_6_2";
  else {
    TRAP;
  }

  Array<DxcDefine> dxc_defines;
  dxc_defines.init();
  TMP_STORAGE_SCOPE;
  ito(num_defines) {
    DxcDefine d;
    d.Name  = towstr_tmp(defines[i].first);
    d.Value = towstr_tmp(defines[i].second);
    dxc_defines.push(d);
  }
  defer(dxc_defines.release());

  WCHAR const *options[] = {L"-spirv"};

  CComPtr<IDxcOperationResult> result;
  HRESULT                      hr = compiler->Compile(blob,                        // pSource
                                 L"shader.hlsl",              // pSourceName
                                 L"main",                     // pEntryPoint
                                 profile,                     // pTargetProfile
                                 options, ARRAYSIZE(options), // pArguments, argCount
                                 dxc_defines.ptr, dxc_defines.size, // pDefines, defineCount
                                 NULL,     // pIncludeHandler
                                 &result); // ppResult
  if (SUCCEEDED(hr)) result->GetStatus(&hr);
  if (FAILED(hr)) {
    if (result) {
      CComPtr<IDxcBlobEncoding> errorsBlob;
      hr = result->GetErrorBuffer(&errorsBlob);
      if (SUCCEEDED(hr) && errorsBlob) {
        fprintf(stdout, "Compilation failed with errors:\n%s\n",
                (const char *)errorsBlob->GetBufferPointer());
      }
    }
    TRAP;
  } else {
    CComPtr<IDxcBlob> spirv;
    result->GetResult(&spirv);
    Array<u32> out;
    out.init((u32 *)spirv->GetBufferPointer(), spirv->GetBufferSize() / 4);
    return out;
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

u32 to_vk_memory_bits(rd::Memory_Type type) {
  u32 prop_flags = 0;
  if (type == rd::Memory_Type::CPU_READ_WRITE) {
    prop_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (type == rd::Memory_Type::CPU_WRITE_GPU_READ) {
    prop_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  if (type == rd::Memory_Type::GPU_LOCAL) {
    prop_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
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
  u32                                 num_bindings;
  VkVertexInputAttributeDescription   attributes[0x10];
  u32                                 num_attributes;
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
        {VK_DESCRIPTOR_TYPE_SAMPLER, 4096},                //
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, //
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096},          //
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},         //
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096},         //
    };
    VkDescriptorPoolCreateInfo info;
    MEMZERO(info);
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = ARRAY_SIZE(aPoolSizes);
    info.pPoolSizes    = aPoolSizes;
    info.maxSets       = 1024;
    info.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
                 VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
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
  void free(VkDescriptorSet set) { vkFreeDescriptorSets(device, pool, 1, &set); }
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
          binding_flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                             VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        } else {
          binding_flags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
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
      set_layout_create_info.flags = 0 | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
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

class VkDeviceContext;

struct VK_Binding_Signature {
  VkPipelineLayout                         pipeline_layout = VK_NULL_HANDLE;
  InlineArray<VkDescriptorSetLayout, 0x10> set_layouts{};
  u32                                      push_constants_size = 0;
  static VK_Binding_Signature *            create(VkDeviceContext *                    dev_ctx,
                                                  rd::Binding_Table_Create_Info const &info);
  void                                     release(VkDeviceContext *dev_ctx);
};

struct VK_Binding_Signature_Slot : public Slot {
  VK_Binding_Signature *signature = NULL;
  VK_Binding_Signature_Slot()     = default;
  VK_Binding_Signature_Slot(VK_Binding_Signature *signature) : signature(signature) {}
};

class VK_Binding_Table : public rd::IBinding_Table {
  private:
  VkDeviceContext *                  dev_ctx   = NULL;
  VK_Binding_Signature *             signature = NULL;
  InlineArray<VkDescriptorSet, 0x10> sets{};
  InlineArray<ID, 0x10>              set_ids{};
  u8                                 push_constants_storage[128]{};

  public:
  u32                      get_num_sets() { return sets.size; }
  VkDescriptorSet          get_set(u32 i) { return sets[i]; }
  VK_Binding_Signature *   get_signature() { return signature; }
  static VK_Binding_Table *create(VkDeviceContext *dev_ctx, VK_Binding_Signature *signature);
  VkDescriptorSet          get_set();
  VkDescriptorSetLayout    get_set_layout();
  void                     bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                                        size_t size) override;
  void                     bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) override;
  void bind_UAV_buffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                       size_t size) override;
  void bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                    rd::Image_Subresource const &range, rd::Format format) override;
  void bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                        rd::Image_Subresource const &range, rd::Format format) override;
  void release() override;
  void push_constants(void const *data, size_t offset, size_t size) override {
    memcpy(push_constants_storage + offset, data, size);
  }
  void bind(VkCommandBuffer cmd, VkPipelineBindPoint bind_point) {
    if (signature->push_constants_size) {
      vkCmdPushConstants(cmd, signature->pipeline_layout, VK_SHADER_STAGE_ALL, 0,
                         signature->push_constants_size, push_constants_storage);
    }
    ito(signature->set_layouts.size) {
      VkDescriptorSet set = sets[i];
      vkCmdBindDescriptorSets(cmd, bind_point, signature->pipeline_layout, 0, 1, &set, 0, NULL);
    }
  }
};

struct Graphics_Pipeline_Wrapper : public Slot {
  // VkPipelineLayout pipeline_layout;
  VkPipeline     pipeline;
  VkShaderModule ps_module;
  VkShaderModule vs_module;
  // u32              push_constants_size;

  void release(VkDevice device) {
    // ito(set_layouts.size) {
    // if (set_layouts[i] != VK_NULL_HANDLE)
    // vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
    //}
    // set_layouts.release();
    // bindings.release();
    // vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, vs_module, NULL);
    vkDestroyShaderModule(device, ps_module, NULL);
    MEMZERO(*this);
  }

  void init(VkDevice              device,                   //
            VK_Binding_Signature *table, Render_Pass &pass, //
            Shader_Info &            vs_shader,             //
            Shader_Info &            ps_shader,             //
            Graphics_Pipeline_State &pipeline_info) {
    MEMZERO(*this);
    (void)pipeline_info;
    // Hash_Set<Pair<u32, u32>> bindings_set;
    // bindings_set.init();
    // defer(bindings_set.release());
    // InlineArray<VkDescriptorSetLayout, 8> set_layouts;
    // MEMZERO(set_layouts);
    // push_constants_size = 0;
    VkPipelineShaderStageCreateInfo stages[2]{};
    // Shader_Reflection::Table<u32, Shader_Reflection::Table<u32, Shader_Descriptor>>
    //    merged_set_table;
    // merged_set_table.init();
    // defer(merged_set_table.release());
    /*ito(num_tables) {
      VK_Binding_Table *vk_table = (VK_Binding_Table *)table[i];
      set_layouts.push(vk_table->get_set_layout());
    }*/
    {
      vs_module = vs_shader.compile(device);
      VkPipelineShaderStageCreateInfo stage;
      MEMZERO(stage);
      stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module = vs_module;
      stage.pName  = "main";
      stages[0]    = stage;
      // Shader_Reflection vs_reflection = reflect_shader(vs_shader);
      // push_constants_size             = MAX(vs_reflection.push_constants_size,
      // push_constants_size);
      // vs_reflection.merge_into(merged_set_table);
      // defer(vs_reflection.release(device));
    }
    {
      ps_module = ps_shader.compile(device);
      VkPipelineShaderStageCreateInfo stage;
      MEMZERO(stage);
      stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
      stage.module = ps_module;
      stage.pName  = "main";
      stages[1]    = stage;
      // Shader_Reflection ps_reflection = reflect_shader(ps_shader);
      // push_constants_size             = MAX(ps_reflection.push_constants_size,
      // push_constants_size);
      // ps_reflection.merge_into(merged_set_table);
      // defer(ps_reflection.release(device));
    }
    /*merged_set_table.iter_pairs(
        [&](u32 set, Shader_Reflection::Table<u32, Shader_Descriptor> &bindings) {
          bindings.iter_pairs([&](u32 index, Shader_Descriptor &binding) {
            bindings_set.insert({set, index});
          });
        });*/
    // Shader_Reflection::create_layouts(device, merged_set_table, set_layouts);

    {
      VkGraphicsPipelineCreateInfo info;
      MEMZERO(info);
      info.sType  = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      info.layout = table->pipeline_layout;

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
        /*  VK_DYNAMIC_STATE_DEPTH_BIAS,
          VK_DYNAMIC_STATE_LINE_WIDTH,*/
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
  // VkPipelineLayout pipeline_layout;
  VkPipeline     pipeline;
  VkShaderModule cs_module;
  // u32              push_constants_size;
  ID   shader_id;
  void release(VkDevice device) {
    /* ito(set_layouts.size) {
       if (set_layouts[i] != VK_NULL_HANDLE)
         vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
     }*/
    // set_layouts.release();
    // vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, cs_module, NULL);
    MEMZERO(*this);
  }

  void init(VkDevice              device, //
            VK_Binding_Signature *table, Shader_Info &cs_shader) {
    MEMZERO(*this);
    // InlineArray<VkDescriptorSetLayout, 8> set_layouts;
    // MEMZERO(set_layouts);
    shader_id = cs_shader.id;
    // Hash_Set<Pair<u32, u32>> bindings_set;
    // bindings_set.init();
    // defer(bindings_set.release());
    // push_constants_size = 0;
    VkPipelineShaderStageCreateInfo stages[1];
    // Shader_Reflection::Table<u32, Shader_Reflection::Table<u32, Shader_Descriptor>>
    // merged_set_table;
    // merged_set_table.init();
    // defer(merged_set_table.release());
    {
      cs_module = cs_shader.compile(device);
      VkPipelineShaderStageCreateInfo stage;
      MEMZERO(stage);
      stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = cs_module;
      stage.pName  = "main";
      stages[0]    = stage;
      // Shader_Reflection cs_reflection = reflect_shader(cs_shader);
      // push_constants_size             = MAX(cs_reflection.push_constants_size,
      // push_constants_size);
      // cs_reflection.merge_into(merged_set_table);
      // defer(cs_reflection.release(device));
    }
    /*merged_set_table.iter_pairs(
        [&](u32 set, Shader_Reflection::Table<u32, Shader_Descriptor> &bindings) {
          bindings.iter_pairs([&](u32 index, Shader_Descriptor &binding) {
            bindings_set.insert({set, index});
          });
        });*/
    // Shader_Reflection::create_layouts(device, merged_set_table, set_layouts);
    /* ito(num_tables) {
       VK_Binding_Table *vk_table = (VK_Binding_Table *)table[i];
       set_layouts.push(vk_table->get_set_layout());
     }
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
     }*/
    {
      VkComputePipelineCreateInfo info;
      MEMZERO(info);
      info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      info.layout = table->pipeline_layout;
      info.stage  = stages[0];
      VK_ASSERT_OK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
    }
  }
};

template <typename T, typename Parent_t> //
class Resource_Array {
  protected:
  Array<T>          items;
  Array<u32>        free_items;
  std::atomic<bool> can_read;
  std::mutex        mutex;
  struct Deferred_Release {
    u32 timer;
    u32 item_index;
  };
  Array<Deferred_Release> limbo_items;

  public:
  void dump() {
    fprintf(stdout, "Resource_Array %s:", Parent_t::NAME);
    fprintf(stdout, "  items: %i", (u32)items.size);
    fprintf(stdout, "  free : %i", (u32)free_items.size);
    fprintf(stdout, "  limbo: %i\n", (u32)limbo_items.size);
  }
  void init() {
    can_read = true;
    items.init();
    free_items.init();
    limbo_items.init();
  }
  void release() {
    can_read = false;
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
    can_read = false;
    defer(can_read = true);
    std::lock_guard<std::mutex> _lock(mutex);
    ASSERT_DEBUG(!id.is_null());
    items[id.index()].disable();
    free_items.push(id.index());
  }
  ID push(T t) {
    can_read = false;
    defer(can_read = true);
    std::lock_guard<std::mutex> _lock(mutex);
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
  /*T &operator[](ID id) {
    ASSERT_DEBUG(!id.is_null() && items[id.index()].get_id().index() == id.index());
    return items[id.index()];
  }*/
  template <typename T> void acquire(ID id, T t) {
    ASSERT_DEBUG(!id.is_null() && items[id.index()].get_id().index() == id.index());
    can_read = false;
    defer(can_read = true);
    std::lock_guard<std::mutex> _lock(mutex);
    t(items[id.index()]);
  }
  // T operator[](ID id) {
  T read(ID id) {
    ASSERT_DEBUG(!id.is_null() && items[id.index()].get_id().index() == id.index());
    if (can_read) return items[id.index()];
    std::lock_guard<std::mutex> _lock(mutex);
    return items[id.index()];
  }
  void add_ref(ID id) {
    can_read = false;
    defer(can_read = true);
    std::lock_guard<std::mutex> _lock(mutex);
    ASSERT_DEBUG(!id.is_null() && items[id.index()].get_id().index() == id.index());
    items[id.index()].frames_referenced++;
  }
  void remove(ID id, u32 timeout) {
    can_read = false;
    defer(can_read = true);
    std::lock_guard<std::mutex> _lock(mutex);
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
    can_read = false;
    defer(can_read = true);
    std::lock_guard<std::mutex> _lock(mutex);
    Array<Deferred_Release>     new_limbo_items;
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

namespace {
static std::atomic<int> g_thread_counter;
static int              get_thread_id() {
  static thread_local int id = g_thread_counter++;
  return id;
};
} // namespace

struct VkDeviceContext {
  static constexpr u32 MAX_SC_IMAGES = 0x10;
  static constexpr u32 MAX_THREADS   = 0x100;

  VkSurfaceKHR surface       = VK_NULL_HANDLE;
  i32          window_width  = 1280;
  i32          window_height = 720;

  VkInstance                 instance                              = VK_NULL_HANDLE;
  VkPhysicalDevice           physdevice                            = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties device_properties                     = {};
  VkQueue                    queue                                 = VK_NULL_HANDLE;
  VkDevice                   device                                = VK_NULL_HANDLE;
  VkCommandPool              cmd_pools[MAX_SC_IMAGES][MAX_THREADS] = {};
  VkQueryPool                query_pool                            = VK_NULL_HANDLE;
  u32                        timestamp_frequency;
  std::atomic<u32>           query_cursor;
  // VkCommandBuffer            cmd_buffers[MAX_SC_IMAGES] = {};

  VkSwapchainKHR     swapchain                = VK_NULL_HANDLE;
  uint32_t           sc_image_count           = 0;
  ID                 sc_images[MAX_SC_IMAGES] = {};
  VkExtent2D         sc_extent                = {};
  VkSurfaceFormatKHR sc_format                = {};

  u32         frame_id                         = 0;
  u32         cmd_index                        = 0;
  u32         image_index                      = 0;
  VkFence     frame_fences[MAX_SC_IMAGES]      = {};
  VkSemaphore sc_free_sem[MAX_SC_IMAGES]       = {};
  VkSemaphore render_finish_sem[MAX_SC_IMAGES] = {};
  // Descriptor_Pool desc_pools[MAX_SC_IMAGES][MAX_THREADS] = {};
  Descriptor_Pool desc_pool = {};

  u32 graphics_queue_id = 0;
  u32 compute_queue_id  = 0;
  u32 transfer_queue_id = 0;

  struct Mem_Chunk_Array : Resource_Array<Mem_Chunk, Mem_Chunk_Array> {
    static constexpr char const NAME[]  = "Mem_Chunk_Array";
    VkDeviceContext *           dev_ctx = NULL;
    std::mutex                  search_mutex;
    void release_item(Mem_Chunk &mem_chunk) { mem_chunk.release(dev_ctx->device); }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
    ID find_mem_chunk(u32 prop_flags, u32 memory_type_bits, u32 alignment, u32 size) {
      can_read = false;
      defer(can_read = true);
      std::lock_guard<std::mutex> _lock(search_mutex);
      (void)alignment;
      ito(items.size) { // look for a suitable memory chunk
        Mem_Chunk &chunk = items[i];
        if ((chunk.prop_flags & prop_flags) == prop_flags &&
            (chunk.memory_type_bits & memory_type_bits) == memory_type_bits) {
          if (chunk.has_space(alignment, size)) {
            return ID{i + 1};
          }
        }
      }
      // if failed create a new one
      Mem_Chunk new_chunk{};
      u32       alloc_size = 1 << 16;
      if (alloc_size < size) {
        alloc_size = (size + alloc_size - 1) & ~(alloc_size - 1);
      }
      new_chunk.init(dev_ctx->device, alloc_size, _find_mem_type(memory_type_bits, prop_flags),
                     prop_flags, memory_type_bits);
      ito(items.size) { // look for a free memory chunk slot
        Mem_Chunk &chunk = items[i];
        if (chunk.is_empty()) {
          chunk = std::move(new_chunk);
          return ID{i + 1};
        }
      }
      ASSERT_DEBUG(new_chunk.has_space(alignment, size));
      return push(new_chunk);
    }
    u32 _find_mem_type(u32 type, VkMemoryPropertyFlags prop_flags) {
      VkPhysicalDeviceMemoryProperties props;
      vkGetPhysicalDeviceMemoryProperties(dev_ctx->physdevice, &props);
      ito(props.memoryTypeCount) {
        if (type & (1 << i) && (props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) {
          return i;
        }
      }
      TRAP;
    }
    void release_unreferenced() {
      ito(items.size) {
        Mem_Chunk &chunk = items[i];
        if (chunk.is_referenced() == false) {
          chunk.release(dev_ctx->device);
        }
      }
    }
  } mem_chunks;
  struct Signature_Array : Resource_Array<VK_Binding_Signature_Slot, Signature_Array> {
    static constexpr char const NAME[]  = "Signature_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(VK_Binding_Signature_Slot &item) {
      item.signature->release(dev_ctx);
      item.signature = NULL;
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } signatures;
  struct Buffer_Array : Resource_Array<Buffer, Buffer_Array> {
    static constexpr char const NAME[]  = "Buffer_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Buffer &buf) {
      vkDestroyBuffer(dev_ctx->device, buf.buffer, NULL);
      dev_ctx->mem_chunks.acquire(buf.mem_chunk_id,
                                  [=](Mem_Chunk &mem_chunk) { mem_chunk.rem_reference(); });
      buf.release();
      MEMZERO(buf);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } buffers;
  struct Image_Array : Resource_Array<Image, Image_Array> {
    static constexpr char const NAME[]  = "Image_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Image &img) {
      if (img.mem_chunk_id.is_null() == false) {
        // True in case of swap chain images
        vkDestroyImage(dev_ctx->device, img.image, NULL);
        dev_ctx->mem_chunks.acquire(img.mem_chunk_id,
                                    [=](Mem_Chunk &mem_chunk) { mem_chunk.rem_reference(); });
      }
      img.release();
      MEMZERO(img);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } images;
  struct BufferView_Array : Resource_Array<BufferView, BufferView_Array> {
    static constexpr char const NAME[]  = "BufferView_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(BufferView &buf) {
      vkDestroyBufferView(dev_ctx->device, buf.view, NULL);
      // dev_ctx->buffers.read(buf.buf_id).rem_reference();
      MEMZERO(buf);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } buffer_views;
  struct ImageView_Array : Resource_Array<ImageView, ImageView_Array> {
    static constexpr char const NAME[]  = "ImageView_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(ImageView &img) {
      vkDestroyImageView(dev_ctx->device, img.view, NULL);
      // dev_ctx->images[img.img_id).rem_reference();
      MEMZERO(img);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } image_views;
  struct Shader_Array : Resource_Array<Shader_Info, Shader_Array> {
    static constexpr char const NAME[]  = "Shader_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Shader_Info &shader) { shader.release(); }
    void                        init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } shaders;
  struct Render_Pass_Array : Resource_Array<Render_Pass, Render_Pass_Array> {
    static constexpr char const NAME[]  = "Render_Pass_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Render_Pass &item) { item.release(dev_ctx->device); }
    void                        init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } render_passes;
  struct Pipe_Array : Resource_Array<Graphics_Pipeline_Wrapper, Pipe_Array> {
    static constexpr char const NAME[]  = "Pipe_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Graphics_Pipeline_Wrapper &item) {
      item.release(dev_ctx->device);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } pipelines;
  struct Compute_Pipe_Array : Resource_Array<Compute_Pipeline_Wrapper, Compute_Pipe_Array> {
    static constexpr char const NAME[]  = "Compute_Pipe_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Compute_Pipeline_Wrapper &item) {
      dev_ctx->shaders.remove(item.shader_id, 0);
      item.release(dev_ctx->device);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } compute_pipelines;
  struct Fence_Array : Resource_Array<Fence, struct Fence_Array> {
    static constexpr char const NAME[]  = "Fence_Pipe_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Fence &item) {
      vkDestroyFence(dev_ctx->device, item.fence, NULL);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } fences;
  struct Event_Array : Resource_Array<Event, struct Event_Array> {
    static constexpr char const NAME[]  = "Event_Pipe_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Event &item) {
      vkDestroyEvent(dev_ctx->device, item.event, NULL);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } events;
  struct CmdBuffer_Array : Resource_Array<CommandBuffer, struct CmdBuffer_Array> {
    static constexpr char const NAME[]  = "CmdBuffer";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(CommandBuffer &item) {
      vkFreeCommandBuffers(dev_ctx->device, item.pool, 1, &item.cmd);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } cmd_buffers;
  struct Sem_Array : Resource_Array<Semaphore, struct Sem_Array> {
    static constexpr char const NAME[]  = "Sem_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Semaphore &item) {
      vkDestroySemaphore(dev_ctx->device, item.sem, NULL);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } semaphores;
  struct Sampler_Array : Resource_Array<Sampler, Sampler_Array> {
    static constexpr char const NAME[]  = "Sampler_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(Sampler &item) {
      vkDestroySampler(dev_ctx->device, item.sampler, NULL);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } samplers;
  struct Set_Array : Resource_Array<DescriptorSet, Set_Array> {
    static constexpr char const NAME[]  = "Set_Array";
    VkDeviceContext *           dev_ctx = NULL;
    void                        release_item(DescriptorSet &item) {
      vkFreeDescriptorSets(dev_ctx->device, dev_ctx->desc_pool.pool, 1, &item.set);
      MEMZERO(item);
    }
    void init(VkDeviceContext *dev_ctx) {
      this->dev_ctx = dev_ctx;
      Resource_Array::init();
    }
  } sets;

  // ID                                          cur_pass;
  Hash_Table<Graphics_Pipeline_State, ID> pipeline_cache;
  Hash_Table<ID, ID>                      compute_pipeline_cache;
  Hash_Table<u64, ID>                     shader_cache;
  // Hash_Table<rd::Render_Pass_Create_Info, ID> pass_cache;
  std::mutex mutex;
  std::mutex mutex2;

  void init_ds() {
    shader_cache.init();
    pipeline_cache.init();
    compute_pipeline_cache.init();
    mem_chunks.init(this);
    signatures.init(this);
    buffers.init(this);
    samplers.init(this);
    images.init(this);
    shaders.init(this);
    buffer_views.init(this);
    image_views.init(this);
    render_passes.init(this);
    sets.init(this);
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
    compute_pipelines.release();
    shaders.release();
    buffer_views.release();
    image_views.release();
    render_passes.release();
    sets.release();
    pipelines.release();
    fences.release();
    events.release();
    cmd_buffers.release();
    semaphores.release();
    // ito(mem_chunks.size) mem_chunks[i].release(device);
    mem_chunks.release();
    signatures.release();
    ito(sc_image_count) jto(MAX_THREADS) if (cmd_pools[i][j] != VK_NULL_HANDLE)
        vkDestroyCommandPool(device, cmd_pools[i][j], NULL);
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, query_pool, NULL);
    ito(sc_image_count) vkDestroySemaphore(device, sc_free_sem[i], NULL);
    ito(sc_image_count) vkDestroySemaphore(device, render_finish_sem[i], NULL);
    ito(sc_image_count) vkDestroyFence(device, frame_fences[i], NULL);
    // ito(sc_image_count) jto(MAX_THREADS) desc_pools[i][j].release();
    desc_pool.release();
    vkDestroySwapchainKHR(device, swapchain, NULL);
    vkDestroyDevice(device, NULL);
    if (surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance, surface, NULL);
    vkDestroyInstance(instance, NULL);
    delete this;
  }

  // Descriptor_Pool &get_descriptor_pool() { return desc_pools[cmd_index][get_thread_id()]; }

  VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) {
    std::lock_guard<std::mutex> _lock(mutex);
    return desc_pool.allocate(layout);
  }
  void free_set(VkDescriptorSet set) {
    std::lock_guard<std::mutex> _lock(mutex);
    desc_pool.free(set);
  }

  u32 allocate_timestamp_id() { return (query_cursor++ % 1000); }

  VkDeviceMemory _alloc_memory(u32 property_flags, VkMemoryRequirements reqs) {
    VkMemoryAllocateInfo info;
    MEMZERO(info);
    info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize  = reqs.size;
    info.memoryTypeIndex = mem_chunks._find_mem_type(reqs.memoryTypeBits, property_flags);
    VkDeviceMemory mem;
    VK_ASSERT_OK(vkAllocateMemory(device, &info, nullptr, &mem));
    return mem;
  }

  Pair<VkBuffer, VkDeviceMemory> create_transient_buffer(u32 size) {
    std::lock_guard<std::mutex> _lock(mutex);
    VkBuffer                    buf;
    VkBufferCreateInfo          cinfo;
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
    VkDeviceMemory mem = _alloc_memory(
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, reqs);
    VK_ASSERT_OK(vkBindBufferMemory(device, buf, mem, 0));
    return {buf, mem};
  }

  Resource_ID create_buffer(rd::Buffer_Create_Info info) {
    std::lock_guard<std::mutex> _lock(mutex);
    u32                         prop_flags = to_vk_memory_bits(info.memory_type);
    VkBuffer                    buf;
    VkBufferCreateInfo          cinfo;
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
    new_buf.buffer      = buf;
    new_buf.create_info = cinfo;
    // new_buf.ref_cnt      = 1;

    ID chunk_index =
        mem_chunks.find_mem_chunk(prop_flags, reqs.memoryTypeBits, reqs.alignment, reqs.size);
    new_buf.mem_chunk_id = chunk_index;
    mem_chunks.acquire(chunk_index, [&](Mem_Chunk &chunk) {
      new_buf.mem_offset = chunk.alloc(reqs.alignment, reqs.size);
      vkBindBufferMemory(device, new_buf.buffer, chunk.mem, new_buf.mem_offset);
    });
    return {buffers.push(new_buf), (i32)Resource_Type::BUFFER};
  }

  Graphics_Pipeline_State convert_graphics_state(rd::Graphics_Pipeline_State state) {
    Graphics_Pipeline_State graphics_state{};
    graphics_state.topology = to_vk(state.topology);
    ito(state.num_vs_bindings) {
      graphics_state.bindings[i].binding = i;
      if (state.bindings[i].inputRate == rd::Input_Rate::VERTEX)
        graphics_state.bindings[i].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      else if (state.bindings[i].inputRate == rd::Input_Rate::INSTANCE)
        graphics_state.bindings[i].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
      graphics_state.bindings[i].stride = state.bindings[i].stride;
    }
    ito(state.num_attributes) {
      auto info                             = state.attributes[i];
      graphics_state.attributes[i].binding  = info.binding;
      graphics_state.attributes[i].format   = to_vk(info.format);
      graphics_state.attributes[i].location = info.location;
      graphics_state.attributes[i].offset   = info.offset;
    }
    graphics_state.vs       = state.vs.id;
    graphics_state.ps       = state.ps.id;
    graphics_state.rs_state = state.rs_state;
    graphics_state.ds_state = state.ds_state;
    graphics_state.ms_state = state.ms_state;
    ito(state.num_rts) {
      auto                                bl = state.blend_states[i];
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

      graphics_state.blend_states[i] = bs;
    }
    graphics_state.num_attributes = state.num_attributes;
    graphics_state.num_bindings   = state.num_vs_bindings;
    graphics_state.num_rts        = state.num_rts;
    return graphics_state;
  }

  void *map_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)Resource_Type::BUFFER);
    Buffer buf  = buffers.read(res_id.id);
    void * data = NULL;
    mem_chunks.acquire(buf.mem_chunk_id, [&](Mem_Chunk &mem_chunk) {
      mem_chunk.lock();
      VK_ASSERT_OK(
          vkMapMemory(device, mem_chunk.mem, buf.mem_offset, buf.create_info.size, 0, &data));
    });
    return data;
  }

  void unmap_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)Resource_Type::BUFFER);
    Buffer buf = buffers.read(res_id.id);
    mem_chunks.acquire(buf.mem_chunk_id, [&](Mem_Chunk &mem_chunk) {
      vkUnmapMemory(device, mem_chunk.mem);
      mem_chunk.unlock();
    });
  }

  VkShaderModule compile_spirv(size_t len, u32 *bytecode) {
    std::lock_guard<std::mutex> _lock(mutex);
    VkShaderModuleCreateInfo    info;
    MEMZERO(info);
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = len;
    info.flags    = 0;
    info.pCode    = bytecode;
    VkShaderModule module;
    VK_ASSERT_OK(vkCreateShaderModule(device, &info, NULL, &module));
    return module;
  }

  VkCommandPool cur_cmd_pool() { return cmd_pools[cmd_index][get_thread_id()]; }

  Resource_ID create_fence(bool signaled) {
    std::lock_guard<std::mutex> _lock(mutex);
    VkFenceCreateInfo           info;
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
    std::lock_guard<std::mutex> _lock(mutex);
    VkEventCreateInfo           info;
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
    cmd.pool = cmd_pools[cmd_index][get_thread_id()];

    VkCommandBufferAllocateInfo info;
    MEMZERO(info);
    info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool        = cmd_pools[cmd_index][get_thread_id()];
    info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VK_ASSERT_OK(vkAllocateCommandBuffers(device, &info, &cmd.cmd));
    ID id = cmd_buffers.push(cmd);
    return {id, (u32)Resource_Type::COMMAND_BUFFER};
  }

  Resource_ID create_semaphore() {
    std::lock_guard<std::mutex> _lock(mutex);
    VkSemaphoreCreateInfo       info;
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
    std::lock_guard<std::mutex> _lock(mutex);
    Image                       img = images.read(res_id);
    ImageView                   img_view;
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
      ImageView view = image_views.read(img.views[i]);
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
                           VkFormat format, u32 usage_flags, rd::Memory_Type mem_type) {
    std::lock_guard<std::mutex> _lock(mutex);
    u32                         prop_flags = to_vk_memory_bits(mem_type);
    VkImage                     image;
    VkImageCreateInfo           cinfo;
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
    // new_image.ref_cnt            = 1;
    if (usage_flags & (u32)rd::Image_Usage_Bits::USAGE_DT)
      new_image.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else
      new_image.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    ID chunk_index =
        mem_chunks.find_mem_chunk(prop_flags, reqs.memoryTypeBits, reqs.alignment, reqs.size);
    new_image.mem_chunk_id = chunk_index;
    mem_chunks.acquire(chunk_index, [&](Mem_Chunk &chunk) {
      new_image.mem_offset = chunk.alloc(reqs.alignment, reqs.size);
      vkBindImageMemory(device, new_image.image, chunk.mem, new_image.mem_offset);
    });

    ID          img_id = images.push(new_image);
    Resource_ID res_id = {img_id, (u32)Resource_Type::IMAGE};
    _image_barrier_sync(new_image, 0, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_IMAGE_LAYOUT_GENERAL);
    return res_id;
  }

  Resource_ID create_shader_raw(rd::Stage_t type, string_ref body,
                                Pair<string_ref, string_ref> *defines, size_t num_defines) {
    std::lock_guard<std::mutex> _lock(mutex);
    u64                         shader_hash = hash_of(body);
    ito(num_defines) { shader_hash ^= hash_of(defines[0].first) ^ hash_of(defines[0].second); }
    if (shader_cache.contains(shader_hash)) {
      return {shader_cache.get(shader_hash), (u32)Resource_Type::SHADER};
    }

    String_Builder sb;
    sb.init();
    defer(sb.release());
    sb.reset();
    sb.putf(R"(
#define u32 uint
#define i32 int
#define f32 float
#define f64 double
#define float2_splat(x)  float2(x, x)
#define float3_splat(x)  float3(x, x, x)
#define float4_splat(x)  float4(x, x, x, x)
)");
    sb.putstr(body);
    // preprocess_shader(sb, body);
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

    si.init(type, shader_hash, compile_hlsl(device, text, kind, defines, num_defines));

    ID shid = shaders.push(si);
    shader_cache.insert(shader_hash, shid);
    return {shid, (u32)Resource_Type::SHADER};
  }
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) {
    std::lock_guard<std::mutex> _lock(mutex);
    Sampler                     sm;
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
    std::lock_guard<std::mutex> _lock(mutex);
    rd::Image2D_Info            info;
    MEMZERO(info);
    Image img   = images.read(sc_images[image_index]);
    info.format = from_vk(img.info.format);
    info.height = img.info.extent.height;
    info.width  = img.info.extent.width;
    info.layers = img.info.arrayLayers;
    info.levels = img.info.mipLevels;
    return info;
  }
  rd::Image_Info get_image_info(Resource_ID res_id) {
    std::lock_guard<std::mutex> _lock(mutex);
    rd::Image_Info              info;
    MEMZERO(info);
    Image img     = images.read(res_id.id);
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
    std::lock_guard<std::mutex> _lock(mutex2);
    if (res_id.type == (u32)Resource_Type::PASS) {
      // Render_Pass pass = render_passes.read(res_id.id);
      render_passes.remove(res_id.id, 3);
      // if (pass_cache.contains(pass.create_info)) pass_cache.remove(pass.create_info);
    } else if (res_id.type == (u32)Resource_Type::SET) {
      sets.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::BUFFER) {
      Buffer buf = buffers.read(res_id.id);
      // buf.rem_reference();
      ito(buf.views.size) buffer_views.remove(buf.views[i], 3);
      buffers.remove(res_id.id, 4);
    } else if (res_id.type == (u32)Resource_Type::BUFFER_VIEW) {
      BufferView view = buffer_views.read(res_id.id);
      Buffer     buf  = buffers.read(view.buf_id);
      buf.views.remove(res_id.id);
      buffer_views.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::IMAGE_VIEW) {
      ImageView view = image_views.read(res_id.id);
      Image     img  = images.read(view.img_id);
      img.views.remove(res_id.id);
      image_views.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::IMAGE) {
      Image img = images.read(res_id.id);
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
    } else if (res_id.type == (u32)Resource_Type::GRAPHICS_PSO) {
      pipelines.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::SIGNATURE) {
      signatures.remove(res_id.id, 3);
    } else if (res_id.type == (u32)Resource_Type::COMPUTE_PSO) {
      compute_pipelines.remove(res_id.id, 3);
    } else {
      TRAP;
    }
  }

  void release_swapchain() {
    if (swapchain != VK_NULL_HANDLE) {
      vkDestroySwapchainKHR(device, swapchain, NULL);
    }
    ito(sc_image_count) {
      Image img = images.read(sc_images[i]);
      // img.rem_reference();
      jto(img.views.size) { image_views.remove(img.views[j], 3); }
      images.remove(sc_images[i], 3);
      sc_images[i] = ID{0};
    }
  }

  void update_swapchain() {
    if (surface == VK_NULL_HANDLE) {
      // In the case when we don't have a swap chain just allocate 1 command pool
      sc_image_count = 1;
      return;
    }
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
      if (present_modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) { // prefer mailbox
        present_mode_of_choice = VK_PRESENT_MODE_IMMEDIATE_KHR;
        break;
      }
    }
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
      _image_barrier_sync(image, 0, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }
  }

  void init(void *window_handler) {
    init_ds();

    TMP_STORAGE_SCOPE;

    const char *instance_extensions[] = {
#ifdef VK_USE_PLATFORM_WIN32_KHR
        VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#else
        VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

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
    info.enabledExtensionCount   = ARRAYSIZE(instance_extensions);
    info.ppEnabledExtensionNames = instance_extensions;

    VK_ASSERT_OK(vkCreateInstance(&info, nullptr, &instance));

#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (window_handler) {
      VkWin32SurfaceCreateInfoKHR createInfo;
      MEMZERO(createInfo);
      createInfo.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
      createInfo.hwnd      = (HWND)window_handler;
      createInfo.hinstance = GetModuleHandle(nullptr);
      VK_ASSERT_OK(vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface));
    }
#else
    if (window_handler) {
      VkXcbSurfaceCreateInfoKHR createInfo;
      MEMZERO(createInfo);
      createInfo.sType      = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
      Ptr2 ptrs             = *(Ptr2 *)window_handler;
      createInfo.window     = (xcb_window_t)ptrs.ptr1;
      createInfo.connection = (xcb_connection_t *)ptrs.ptr2;
      VK_ASSERT_OK(vkCreateXcbSurfaceKHR(instance, &createInfo, nullptr, &surface));
    }
#endif

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

          VkBool32 sup = VK_TRUE;
          if (surface != VK_NULL_HANDLE)
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
    ASSERT_DEBUG(pd_index_features.descriptorBindingUniformBufferUpdateAfterBind);
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

      ito(sc_image_count) {
        jto(MAX_THREADS) { VK_ASSERT_OK(vkCreateCommandPool(device, &info, 0, &cmd_pools[i][j])); }
      }
    }

    ito(sc_image_count) {
      if (sc_images[i].is_null()) continue;
      Image image = images.read(sc_images[i]);
      _image_barrier_sync(image, 0, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
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
    desc_pool.init(device);
    // ito(sc_image_count) jto(MAX_THREADS) desc_pools[i][j].init(device);
  }

  void update_surface_size() {
    if (surface != VK_NULL_HANDLE) {
      VkSurfaceCapabilitiesKHR surface_capabilities;
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdevice, surface, &surface_capabilities);
      window_width  = surface_capabilities.currentExtent.width;
      window_height = surface_capabilities.currentExtent.height;
    }
  }

  void _image_barrier_sync(Image const &img, VkAccessFlags old_mem_access,
                           VkImageLayout old_image_layout, VkAccessFlags new_mem_access,
                           VkImageLayout new_image_layout) {
    if (cmd_pools[cmd_index][get_thread_id()] == VK_NULL_HANDLE) return;
    Resource_ID              cmd_id = create_command_buffer();
    CommandBuffer            cmd    = cmd_buffers.read(cmd_id.id);
    VkCommandBufferBeginInfo begin_info;
    MEMZERO(begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd.cmd, &begin_info);
    {
      VkImageMemoryBarrier bar;
      MEMZERO(bar);
      bar.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bar.srcAccessMask                   = old_mem_access;
      bar.dstAccessMask                   = new_mem_access;
      bar.oldLayout                       = old_image_layout;
      bar.newLayout                       = new_image_layout;
      bar.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
      bar.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
      bar.image                           = img.image;
      bar.subresourceRange.aspectMask     = img.aspect;
      bar.subresourceRange.baseArrayLayer = 0;
      bar.subresourceRange.baseMipLevel   = 0;
      bar.subresourceRange.layerCount     = img.info.arrayLayers;
      bar.subresourceRange.levelCount     = img.info.mipLevels;
      vkCmdPipelineBarrier(cmd.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
    }

    vkEndCommandBuffer(cmd.cmd);
    release_resource(cmd_id);
    VkPipelineStageFlags stage_flags[]{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    VkSubmitInfo         submit_info;
    MEMZERO(submit_info);
    submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers    = &cmd.cmd;
    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vkDeviceWaitIdle(device);
  }

  Resource_ID _image_barrier_async(Image const &img, VkAccessFlags old_mem_access,
                                   VkImageLayout old_image_layout, VkAccessFlags new_mem_access,
                                   VkImageLayout new_image_layout) {
    if (cmd_pools[cmd_index] == VK_NULL_HANDLE) return {0u};
    Resource_ID              cmd_id = create_command_buffer();
    CommandBuffer            cmd    = cmd_buffers.read(cmd_id.id);
    VkCommandBufferBeginInfo begin_info;
    MEMZERO(begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd.cmd, &begin_info);
    {
      VkImageMemoryBarrier bar;
      MEMZERO(bar);
      bar.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bar.srcAccessMask                   = old_mem_access;
      bar.dstAccessMask                   = new_mem_access;
      bar.oldLayout                       = old_image_layout;
      bar.newLayout                       = new_image_layout;
      bar.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
      bar.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
      bar.image                           = img.image;
      bar.subresourceRange.aspectMask     = img.aspect;
      bar.subresourceRange.baseArrayLayer = 0;
      bar.subresourceRange.baseMipLevel   = 0;
      bar.subresourceRange.layerCount     = img.info.arrayLayers;
      bar.subresourceRange.levelCount     = img.info.mipLevels;
      vkCmdPipelineBarrier(cmd.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
    }

    vkEndCommandBuffer(cmd.cmd);
    release_resource(cmd_id);
    VkPipelineStageFlags stage_flags[]{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    VkSubmitInfo         submit_info;
    MEMZERO(submit_info);
    Resource_ID sem                  = create_semaphore();
    VkSemaphore raw_sem              = semaphores.read(sem.id).sem;
    submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount   = 1;
    submit_info.pCommandBuffers      = &cmd.cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores    = &raw_sem;
    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    return sem;
  }

  Resource_ID start_frame() {
    // vkDeviceWaitIdle(device);
    // Sleep(1000);
    buffers.tick();
    samplers.tick();
    images.tick();
    shaders.tick();
    buffer_views.tick();
    image_views.tick();
    render_passes.tick();
    sets.tick();
    pipelines.tick();
    signatures.tick();
    compute_pipelines.tick();
    fences.tick();
    events.tick();
    cmd_buffers.tick();
    semaphores.tick();
    mem_chunks.tick();
    mem_chunks.release_unreferenced();
    Array<Resource_ID> resources_to_remove;
    resources_to_remove.init();
    defer(resources_to_remove.release());
    /* render_passes.for_each([&](Render_Pass const &p) {
       if (p.frames_referenced < -2) {
         resources_to_remove.push({p.id, (u32)Resource_Type::PASS});
       }
     });*/
    ito(resources_to_remove.size) release_resource(resources_to_remove[i]);

  restart:
    cmd_index = (frame_id++) % sc_image_count;

    if (surface != VK_NULL_HANDLE) {
      update_surface_size();
      if (window_width != (i32)sc_extent.width || window_height != (i32)sc_extent.height) {
        update_swapchain();
      }
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
      jto(MAX_THREADS) VK_ASSERT_OK(vkResetCommandPool(
          device, cmd_pools[cmd_index][j], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
      // jto(MAX_THREADS) desc_pools[cmd_index][j].reset();
      if (1) {

        Image       img = images.read(sc_images[image_index]);
        Resource_ID sem = _image_barrier_async(img, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0,
                                               VK_IMAGE_LAYOUT_GENERAL);
        return sem;
      }
    } else {
      jto(MAX_THREADS) VK_ASSERT_OK(vkResetCommandPool(
          device, cmd_pools[cmd_index][j], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
      // jto(MAX_THREADS) desc_pools[cmd_index][j].reset();
    }
    return {0u};
  }
  void end_frame(VkSemaphore *wait_sem) {
    if (surface != VK_NULL_HANDLE) {
      Resource_ID   cmd_id = create_command_buffer();
      CommandBuffer cmd    = cmd_buffers.read(cmd_id.id);
      // vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
      VkCommandBufferBeginInfo begin_info;
      MEMZERO(begin_info);
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cmd.cmd, &begin_info);

      Image img = images.read(sc_images[image_index]);
      {
        VkImageMemoryBarrier bar;
        MEMZERO(bar);
        bar.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar.srcAccessMask                   = 0;
        bar.dstAccessMask                   = 0;
        bar.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
        bar.newLayout                       = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        bar.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        bar.image                           = img.image;
        bar.subresourceRange.aspectMask     = img.aspect;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.baseMipLevel   = 0;
        bar.subresourceRange.layerCount     = img.info.arrayLayers;
        bar.subresourceRange.levelCount     = img.info.mipLevels;
        vkCmdPipelineBarrier(cmd.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
      }

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

class Vk_Ctx : public rd::ICtx {
  private:
  rd::Pass_t        type{};
  VkDeviceContext * dev_ctx;
  ID                graphics_pso{};
  ID                cs{};
  Resource_ID       cmd_id{};
  VkCommandBuffer   cmd{};
  ID                cur_pass{};
  VK_Binding_Table *binding_table = NULL;

  template <typename K, typename V> using Table = Hash_Table<K, V, Default_Allocator, 64>;
  Table<Resource_ID, BufferLaoutTracker> buffer_layouts;
  Table<Resource_ID, ImageLayoutTracker> image_layouts;
  Pool<u8>                               tmp_pool;

  VkImageLayout _get_image_layout(Resource_ID img_id) {
    if (!image_layouts.contains(img_id)) {
      return VK_IMAGE_LAYOUT_GENERAL;
    }
    ImageLayoutTracker &il = image_layouts.get(img_id);
    return il.layout;
  }

  void _buffer_barrier(VkCommandBuffer cmd, Resource_ID buf_id, VkAccessFlags bits) {
    if (!buffer_layouts.contains(buf_id)) {
      buffer_layouts.insert(buf_id, BufferLaoutTracker{});
    }
    BufferLaoutTracker &bt     = buffer_layouts.get(buf_id);
    Buffer              buffer = dev_ctx->buffers.read(buf_id.id);
    bt.barrier(cmd, &buffer, bits);
  }

  void _image_barrier(VkCommandBuffer cmd, Resource_ID img_id, VkAccessFlags bits,
                      VkImageLayout layout) {
    if (!image_layouts.contains(img_id)) {
      image_layouts.insert(img_id, ImageLayoutTracker{});
    }
    ImageLayoutTracker &il  = image_layouts.get(img_id);
    Image               img = dev_ctx->images.read(img_id.id);
    il.barrier(cmd, &img, bits, layout);
  }

  void _copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                    size_t dst_offset, u32 size) {
    Buffer       src_buffer = dev_ctx->buffers.read(src_buf_id.id);
    Buffer       dst_buffer = dev_ctx->buffers.read(dst_buf_id.id);
    VkBufferCopy info;
    MEMZERO(info);
    info.dstOffset = dst_offset;
    info.size      = size;
    info.srcOffset = src_offset;
    vkCmdCopyBuffer(cmd, src_buffer.buffer, dst_buffer.buffer, 1, &info);
  }

  void _copy_image_buffer(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                          rd::Image_Copy const &dst_info) {
    Buffer            buffer = dev_ctx->buffers.read(buf_id.id);
    Image             image  = dev_ctx->images.read(img_id.id);
    VkBufferImageCopy info;
    MEMZERO(info);
    info.bufferOffset = offset;
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
    if (dst_info.buffer_row_pitch)
      info.bufferRowLength = dst_info.buffer_row_pitch / get_format_size(image.info.format);
    info.imageOffset = {(i32)dst_info.offset_x, (i32)dst_info.offset_y, (i32)dst_info.offset_z};
    VkImageSubresourceLayers subres;
    MEMZERO(subres);
    subres.aspectMask      = image.aspect;
    subres.baseArrayLayer  = dst_info.layer;
    subres.layerCount      = 1;
    subres.mipLevel        = dst_info.level;
    info.imageSubresource  = subres;
    info.bufferImageHeight = image.info.extent.height;
    vkCmdCopyImageToBuffer(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1,
                           &info);
  }

  void _copy_buffer_image(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                          rd::Image_Copy const &dst_info) {
    Buffer            buffer = dev_ctx->buffers.read(buf_id.id);
    Image             image  = dev_ctx->images.read(img_id.id);
    VkBufferImageCopy info;
    MEMZERO(info);
    info.bufferOffset = offset;
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
    if (dst_info.buffer_row_pitch)
      info.bufferRowLength = dst_info.buffer_row_pitch / get_format_size(image.info.format);
    info.imageOffset = {(i32)dst_info.offset_x, (i32)dst_info.offset_y, (i32)dst_info.offset_z};
    VkImageSubresourceLayers subres;
    MEMZERO(subres);
    subres.aspectMask      = image.aspect;
    subres.baseArrayLayer  = dst_info.layer;
    subres.layerCount      = 1;
    subres.mipLevel        = dst_info.level;
    info.imageSubresource  = subres;
    info.bufferImageHeight = image.info.extent.height;
    vkCmdCopyBufferToImage(cmd, buffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &info);
  }

  void _bind_sets() {
    if (binding_table) {
      binding_table->bind(cmd, type == rd::Pass_t::RENDER ? VK_PIPELINE_BIND_POINT_GRAPHICS
                                                          : VK_PIPELINE_BIND_POINT_COMPUTE);
    }
  }

  public:
  void start_render_pass() override {
    Render_Pass pass = dev_ctx->render_passes.read(cur_pass);
    ito(pass.create_info.rts.size) {
      _image_barrier(cmd, pass.create_info.rts[i].image, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    if (pass.depth_target_view.is_null() == false) {
      _image_barrier(cmd, pass.create_info.depth_target.image,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
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
    vkCmdBeginRenderPass(cmd, &binfo, VK_SUBPASS_CONTENTS_INLINE);
  }
  void end_render_pass() override {
    vkCmdEndRenderPass(cmd);
    Render_Pass pass = dev_ctx->render_passes.read(cur_pass);
    ito(pass.create_info.rts.size) {
      _image_barrier(cmd, pass.create_info.rts[i].image, 0, VK_IMAGE_LAYOUT_GENERAL);
    }
    if (pass.depth_target_view.is_null() == false) {
      _image_barrier(cmd, pass.create_info.depth_target.image, 0, VK_IMAGE_LAYOUT_GENERAL);
    }
  }

  // u64  get_dt() { return last_ms; }
  void reset() {
    vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    this->cur_pass = {0};
    // clear_state();
  }

  void init(rd::Pass_t type, VkDeviceContext *dev_ctx, ID pass_id) {
    this->type    = type;
    this->dev_ctx = dev_ctx;
    buffer_layouts.reset();
    image_layouts.reset();
    cmd_id         = dev_ctx->create_command_buffer();
    cmd            = dev_ctx->cmd_buffers.read(cmd_id.id).cmd;
    tmp_pool       = Pool<u8>::create(1 << 20);
    this->cur_pass = pass_id;
    VkCommandBufferBeginInfo begin_info;
    MEMZERO(begin_info);
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_ASSERT_OK(vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
    VK_ASSERT_OK(vkBeginCommandBuffer(cmd, &begin_info));
  }

  ID submit(VkFence finish_fence, VkSemaphore *wait_sem) {
    ID finish_sem;
    finish_sem = dev_ctx->create_semaphore().id;
    // std::lock_guard<std::mutex> _lock(dev_ctx->mutex);
    buffer_layouts.iter_pairs(
        [=](Resource_ID res_id, BufferLaoutTracker &lt) { _buffer_barrier(cmd, res_id, 0); });
    image_layouts.iter_pairs([=](Resource_ID res_id, ImageLayoutTracker &lt) {
      _image_barrier(cmd, res_id, 0, VK_IMAGE_LAYOUT_GENERAL);
    });
    buffer_layouts.release();
    image_layouts.release();
    vkEndCommandBuffer(cmd);
    VkPipelineStageFlags stage_flags[]{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
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

    VkSemaphore raw_sem           = dev_ctx->semaphores.read(finish_sem).sem;
    submit_info.pSignalSemaphores = &raw_sem;
    vkQueueSubmit(dev_ctx->queue, 1, &submit_info, finish_fence);
    return finish_sem;
  }
  void release() {
    tmp_pool.release();
    dev_ctx->release_resource(cmd_id);
    delete this;
  }
  // CTX
  void bind_vertex_buffer(u32 index, Resource_ID buffer, size_t offset) override {
    Buffer buf = dev_ctx->buffers.read(buffer.id);
    vkCmdBindVertexBuffers(cmd, index, 1, &buf.buffer, &offset);
  }
  void bind_index_buffer(Resource_ID id, u32 offset, rd::Index_t format) override {
    Buffer      buf = dev_ctx->buffers.read(id.id);
    VkIndexType type;
    switch (format) {
    case rd::Index_t::UINT32: type = VK_INDEX_TYPE_UINT32; break;
    case rd::Index_t::UINT16: type = VK_INDEX_TYPE_UINT16; break;
    default: TRAP;
    }
    vkCmdBindIndexBuffer(cmd, buf.buffer, offset, type);
  }
  void bind_compute(Resource_ID id) override {
    Compute_Pipeline_Wrapper gw = dev_ctx->compute_pipelines.read(id.id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gw.pipeline);
    cs = id.id;
  }
  void bind_graphics_pso(Resource_ID pso) override {
    Graphics_Pipeline_Wrapper gw = dev_ctx->pipelines.read(pso.id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gw.pipeline);
    graphics_pso = pso.id;
  }
  void bind_table(rd::IBinding_Table *table) override {
    VK_Binding_Table *vk_table = (VK_Binding_Table *)table;
    binding_table              = vk_table;
  }
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
    vkCmdSetViewport(cmd, 0, 1, &viewports[0]);
  }
  void set_scissor(u32 x, u32 y, u32 width, u32 height) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    VkRect2D scissors[1];
    scissors[0].offset.x      = x;
    scissors[0].offset.y      = y;
    scissors[0].extent.width  = width;
    scissors[0].extent.height = height;
    vkCmdSetScissor(cmd, 0, 1, &scissors[0]);
  }
  void insert_event(Resource_ID event_id) override {
    auto event = dev_ctx->events.read(event_id.id).event;
    vkCmdSetEvent(cmd, event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
  void insert_timestamp(Resource_ID id) override {
    auto pool = dev_ctx->query_pool;
    vkResetQueryPool(dev_ctx->device, pool, id.id.index(), 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, pool, id.id.index());
  }
  void copy_image_to_buffer(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    _copy_image_buffer(buf_id, offset, img_id, dst_info);
  }
  void copy_buffer_to_image(Resource_ID buf_id, size_t offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    _copy_buffer_image(buf_id, offset, img_id, dst_info);
  }
  void copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                   size_t dst_offset, u32 size) override {
    _copy_buffer(src_buf_id, src_offset, dst_buf_id, dst_offset, size);
  }
  void draw_indexed(u32 index_count, u32 instance_count, u32 first_index, u32 first_instance,
                    i32 vertex_offset) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    _bind_sets();
    vkCmdDrawIndexed(cmd, index_count, instance_count, first_index, vertex_offset, first_instance);
  }

  void draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance) override {
    ASSERT_ALWAYS(type == rd::Pass_t::RENDER);
    _bind_sets();
    vkCmdDraw(cmd, vertex_count, instance_count, first_vertex, first_instance);
  }
  void multi_draw_indexed_indirect(Resource_ID arg_buf_id, u32 arg_buf_offset,
                                   Resource_ID cnt_buf_id, u32 cnt_buf_offset, u32 max_count,
                                   u32 stride) override {
    Buffer arg_buf = dev_ctx->buffers.read(arg_buf_id.id);
    Buffer cnt_buf = dev_ctx->buffers.read(cnt_buf_id.id);
    _bind_sets();
    vkCmdDrawIndexedIndirectCount(cmd, arg_buf.buffer, arg_buf_offset, cnt_buf.buffer,
                                  cnt_buf_offset, max_count, stride);
  }
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    _bind_sets();
    vkCmdDispatch(cmd, dim_x, dim_y, dim_z);
  }
  virtual void image_barrier(Resource_ID image_id, rd::Image_Access access) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    VkAccessFlags bits{};
    VkImageLayout layout{};
    to_vk(access, bits, layout);
    _image_barrier(cmd, image_id, bits, layout);
  }
  virtual void buffer_barrier(Resource_ID buf_id, rd::Buffer_Access access) override {
    ASSERT_ALWAYS(type == rd::Pass_t::COMPUTE);
    _buffer_barrier(cmd, buf_id, to_vk(access));
  }
  void fill_buffer(Resource_ID id, size_t offset, size_t size, u32 value) override {
    Buffer buf = dev_ctx->buffers.read(id.id);
    vkCmdFillBuffer(cmd, buf.buffer, offset, size, value);
  }
  void clear_image(Resource_ID id, rd::Image_Subresource const &range,
                   rd::Clear_Value const &cv) override {

    Image             img = dev_ctx->images.read(id.id);
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
  void update_buffer(Resource_ID buf_id, size_t offset, void const *data,
                     size_t data_size) override {
    Buffer buf = dev_ctx->buffers.read(buf_id.id);
    vkCmdUpdateBuffer(cmd, buf.buffer, offset, data_size, data);
  }
  //void RS_set_line_width(float width) override { vkCmdSetLineWidth(cmd, width); }
  //void RS_set_depth_bias(float b) override { vkCmdSetDepthBias(cmd, b, 0.0f, 0.0f); }
}; // namespace

class VkFactory : public rd::IDevice {
  VkDeviceContext *  dev_ctx;
  Array<Resource_ID> release_queue;
  ID                 last_sem;
  std::mutex         mutex;

  void on_frame_begin() { last_sem = {0}; }
  void on_frame_end() {
    if (last_sem.is_null() == false) {
      dev_ctx->release_resource({last_sem, (u32)Resource_Type::SEMAPHORE});
    }
    last_sem = {0};
    ito(release_queue.size) dev_ctx->release_resource(release_queue[i]);
    release_queue.release();
  }

  public:
  VkFactory(void *window_handler) {
    dev_ctx = new VkDeviceContext();
    dev_ctx->init(window_handler);
    last_sem = {0};
    release_queue.init();
  }
  void release() {
    release_queue.release();
    dev_ctx->release();
    delete this;
  }

  bool get_timestamp_state(Resource_ID t0) override {
    std::lock_guard<std::mutex> _lock(mutex);
    u64                         timestamp_results[1];
    VkResult res = vkGetQueryPoolResults(dev_ctx->device, dev_ctx->query_pool, t0.id._id, 1, 8,
                                         timestamp_results, 8, VK_QUERY_RESULT_64_BIT);
    return res == VK_SUCCESS;
  }
  double get_timestamp_ms(Resource_ID t0, Resource_ID t1) override {
    std::lock_guard<std::mutex> _lock(mutex);
    u64                         timestamp_results[2];
    VkResult res0 = vkGetQueryPoolResults(dev_ctx->device, dev_ctx->query_pool, t0.id._id, 1, 8,
                                          timestamp_results + 0, 8, VK_QUERY_RESULT_64_BIT);
    VkResult res1 = vkGetQueryPoolResults(dev_ctx->device, dev_ctx->query_pool, t1.id._id, 1, 8,
                                          timestamp_results + 1, 8, VK_QUERY_RESULT_64_BIT);
    if (res0 == VK_SUCCESS && res1 == VK_SUCCESS) {
      u64 diff = timestamp_results[1] - timestamp_results[0];
      return (double(diff) * dev_ctx->device_properties.limits.timestampPeriod) * 1.0e-6;
    }
    return 0.0;
  }
  Resource_ID create_image(rd::Image_Create_Info info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->create_image(info.width, info.height, info.depth, info.layers, info.levels,
                                 to_vk(info.format), info.usage_bits, rd::Memory_Type::GPU_LOCAL);
  }
  Resource_ID create_graphics_pso(Resource_ID signature, Resource_ID render_pass,
                                  rd::Graphics_Pipeline_State const &state) override {
    VK_Binding_Signature *    sig            = dev_ctx->signatures.read(signature.id).signature;
    Graphics_Pipeline_State   graphics_state = dev_ctx->convert_graphics_state(state);
    Graphics_Pipeline_Wrapper gw;
    ASSERT_DEBUG(!graphics_state.ps.is_null());
    ASSERT_DEBUG(!graphics_state.vs.is_null());
    Shader_Info ps   = dev_ctx->shaders.read(graphics_state.ps);
    Shader_Info vs   = dev_ctx->shaders.read(graphics_state.vs);
    Render_Pass pass = dev_ctx->render_passes.read(render_pass.id);
    gw.init(dev_ctx->device, sig, pass, vs, ps, graphics_state);
    return {dev_ctx->pipelines.push(gw), (u32)Resource_Type::GRAPHICS_PSO};
  }
  Resource_ID create_compute_pso(Resource_ID signature, Resource_ID cs) override {
    Compute_Pipeline_Wrapper gw;
    VK_Binding_Signature *   sig     = dev_ctx->signatures.read(signature.id).signature;
    Shader_Info              cs_info = dev_ctx->shaders.read(cs.id);
    gw.init(dev_ctx->device, sig, cs_info);
    ID pipe_id = dev_ctx->compute_pipelines.push(gw);
    return {pipe_id, (u32)Resource_Type::COMPUTE_PSO};
  }
  Resource_ID create_buffer(rd::Buffer_Create_Info info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->create_buffer(info);
  }
  bool get_event_state(Resource_ID fence_id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    Event                       event    = dev_ctx->events.read(fence_id.id);
    VkResult                    wait_res = vkGetEventStatus(dev_ctx->device, event.event);
    ASSERT_DEBUG(wait_res == VK_EVENT_SET || wait_res == VK_EVENT_RESET);
    return wait_res == VK_EVENT_SET;
  }
  Resource_ID create_shader(rd::Stage_t type, string_ref body,
                            Pair<string_ref, string_ref> *defines, size_t num_defines) override {
    std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->create_shader_raw(type, body, defines, num_defines);
  }
  void *map_buffer(Resource_ID res_id) override {
    // std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->map_buffer(res_id);
  }
  void unmap_buffer(Resource_ID res_id) override {
    // std::lock_guard<std::mutex> _lock(mutex);
    dev_ctx->unmap_buffer(res_id);
  }
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->create_sampler(info);
  }
  void release_resource(Resource_ID id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    release_queue.push(id);
  }
  u32         get_num_swapchain_images() override { return dev_ctx->sc_image_count; }
  Resource_ID get_swapchain_image() override {
    std::lock_guard<std::mutex> _lock(mutex);
    return {dev_ctx->sc_images[dev_ctx->image_index], (u32)Resource_Type::IMAGE};
  }
  rd::Image2D_Info get_swapchain_image_info() override {
    std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->get_swapchain_image_info();
  }
  rd::Image_Info get_image_info(Resource_ID res_id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    return dev_ctx->get_image_info(res_id);
  }
  Resource_ID create_render_pass(rd::Render_Pass_Create_Info const &info) override {
    Render_Pass rp{};
    rp.init();
    rp.create_info          = info;
    u32 depth_attachment_id = 0;
    ito(info.rts.size) {
      VkFormat format;
      if (info.rts[i].format == rd::Format::NATIVE)
        format = dev_ctx->images.read(info.rts[i].image.id).info.format;
      else
        format = to_vk(info.rts[i].format);
      rp.rts_views.push(dev_ctx
                            ->create_image_view(info.rts[i].image.id, //
                                                info.rts[i].level, 1, //
                                                info.rts[i].layer, 1, //
                                                format)
                            .id);
    }
    if (info.depth_target.image.is_null() == false) {
      VkFormat format;
      if (info.depth_target.format == rd::Format::NATIVE)
        format = dev_ctx->images.read(info.depth_target.image.id).info.format;
      else
        format = to_vk(info.depth_target.format);
      rp.depth_target_view = dev_ctx
                                 ->create_image_view(info.depth_target.image.id, //
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
    u32 width  = 0;
    u32 height = 0;
    ito(info.rts.size) {
      VkAttachmentDescription attachment;
      MEMZERO(attachment);
      Image img = dev_ctx->images.read(info.rts[i].image.id);
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
      Image img = dev_ctx->images.read(info.depth_target.image.id);
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

    VK_ASSERT_OK(vkCreateRenderPass(dev_ctx->device, &cinfo, NULL, &rp.pass));
    {
      InlineArray<VkImageView, 8> views;
      views.init();
      defer(views.release());
      ito(rp.rts_views.size) { views.push(dev_ctx->image_views.read(rp.rts_views[i]).view); }
      if (info.depth_target.image.is_null() == false) {
        views.push(dev_ctx->image_views.read(rp.depth_target_view).view);
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
      VK_ASSERT_OK(vkCreateFramebuffer(dev_ctx->device, &info, NULL, &rp.fb));
    }
    return {dev_ctx->render_passes.push(rp), (u32)Resource_Type::PASS};
  }
  rd::ICtx *start_render_pass(Resource_ID render_pass) override {
    std::lock_guard<std::mutex> _lock(mutex);
    Vk_Ctx *                    ctx = new Vk_Ctx();
    ctx->init(rd::Pass_t::RENDER, dev_ctx, render_pass.id);
    return ctx;
  }
  void end_render_pass(rd::ICtx *_ctx) override {
    std::lock_guard<std::mutex> _lock(mutex);
    Vk_Ctx *                    ctx = (Vk_Ctx *)_ctx;
    Resource_ID                 f   = dev_ctx->create_fence(false);
    ID                          finish_sem;
    if (last_sem.is_null() == false) {
      Semaphore s = dev_ctx->semaphores.read(last_sem);
      finish_sem  = ctx->submit(dev_ctx->fences.read(f.id).fence, &s.sem);
      dev_ctx->release_resource({last_sem, (u32)Resource_Type::SEMAPHORE});
      last_sem = {0};
    } else {
      finish_sem = ctx->submit(dev_ctx->fences.read(f.id).fence, NULL);
    }
    last_sem = finish_sem;
    dev_ctx->release_resource(f);
    ctx->release();
  }
  rd::ICtx *start_compute_pass() override {
    std::lock_guard<std::mutex> _lock(mutex);
    Vk_Ctx *                    ctx = new Vk_Ctx();
    ctx->init(rd::Pass_t::COMPUTE, dev_ctx, {0});
    return ctx;
  }
  void wait_idle() {
    std::lock_guard<std::mutex> _lock(mutex);
    vkDeviceWaitIdle(dev_ctx->device);
  }
  Resource_ID create_event() override { return dev_ctx->create_event(); }
  Resource_ID create_timestamp() override {
    return {dev_ctx->allocate_timestamp_id(), (u32)Resource_Type::TIMESTAMP};
  }
  Resource_ID create_signature(rd::Binding_Table_Create_Info const &infos) override {
    return {dev_ctx->signatures.push({VK_Binding_Signature::create(dev_ctx, infos)}),
            (u32)Resource_Type::SIGNATURE};
  }
  rd::IBinding_Table *create_binding_table(Resource_ID signature) override {
    VK_Binding_Signature *sig = dev_ctx->signatures.read(signature.id).signature;
    return VK_Binding_Table::create(dev_ctx, sig);
  }
  void end_compute_pass(rd::ICtx *_ctx) override {
    std::lock_guard<std::mutex> _lock(mutex);
    Vk_Ctx *                    ctx = (Vk_Ctx *)_ctx;
    Resource_ID                 f   = dev_ctx->create_fence(false);

    ID finish_sem;
    if (last_sem.is_valid()) {
      Semaphore s = dev_ctx->semaphores.read(last_sem);
      finish_sem  = ctx->submit(dev_ctx->fences.read(f.id).fence, &s.sem);
      dev_ctx->release_resource({last_sem, (u32)Resource_Type::SEMAPHORE});
      last_sem = {0};
    } else {
      finish_sem = ctx->submit(dev_ctx->fences.read(f.id).fence, NULL);
    }
    last_sem = finish_sem;

    dev_ctx->release_resource(f);
    ctx->release();
  }
  rd::Impl_t getImplType() override { return rd::Impl_t::VULKAN; }
  void       start_frame() override {
    std::lock_guard<std::mutex> _lock(mutex);
    Resource_ID                 sem = dev_ctx->start_frame();
    on_frame_begin();
    last_sem = sem.id;
  }
  void end_frame() override {
    std::lock_guard<std::mutex> _lock(mutex);
    on_frame_end();
    VkSemaphore sem = VK_NULL_HANDLE;
    if (last_sem.is_null() == false) {
      sem = dev_ctx->semaphores.read(last_sem).sem;
    }
    if (sem != VK_NULL_HANDLE)
      dev_ctx->end_frame(&sem);
    else
      dev_ctx->end_frame(NULL);
  }
};

//////////////////////////////////
/// VK_Binding_Table

VK_Binding_Signature *
VK_Binding_Signature::create(VkDeviceContext *                    dev_ctx,
                             rd::Binding_Table_Create_Info const &table_info) {
  VK_Binding_Signature *out = new VK_Binding_Signature;
  out->push_constants_size  = table_info.push_constants_size;
  jto(table_info.spaces.size) {
    VkDescriptorBindingFlags     binding_flags[rd::Binding_Space_Create_Info::MAX_BINDINGS];
    u32                          num_bindings = 0;
    VkDescriptorSetLayoutBinding set_bindings[rd::Binding_Space_Create_Info::MAX_BINDINGS];
    auto const &                 info = table_info.spaces[j];
    ito(info.bindings.size) {
      auto                         dinfo = info.bindings[i];
      VkDescriptorSetLayoutBinding binding_info;
      MEMZERO(binding_info);
      binding_info.binding            = i;
      binding_info.descriptorCount    = dinfo.num_array_elems;
      binding_info.descriptorType     = to_vk(dinfo.type);
      binding_info.pImmutableSamplers = NULL;
      binding_info.stageFlags         = VK_SHADER_STAGE_ALL;
      set_bindings[num_bindings++]    = binding_info;
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_infos;

    ito(num_bindings) {
      if (set_bindings[i].descriptorCount > 1) {
        binding_flags[i] = 0 //
                           | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
            // | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
            ;
      } else {
        binding_flags[i] = 0 //
                             // | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                             // | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
            ;
      }
    }
    ASSERT_DEBUG(num_bindings < rd::Binding_Space_Create_Info::MAX_BINDINGS);
    binding_infos.bindingCount  = num_bindings;
    binding_infos.pBindingFlags = &binding_flags[0];
    binding_infos.pNext         = NULL;
    binding_infos.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

    VkDescriptorSetLayoutCreateInfo set_layout_create_info;
    MEMZERO(set_layout_create_info);
    set_layout_create_info.bindingCount = num_bindings;
    set_layout_create_info.pBindings    = &set_bindings[0];
    set_layout_create_info.flags = 0 | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_create_info.pNext = (void *)&binding_infos;
    VkDescriptorSetLayout set_layout;
    VK_ASSERT_OK(
        vkCreateDescriptorSetLayout(dev_ctx->device, &set_layout_create_info, NULL, &set_layout));
    out->set_layouts.push(set_layout);
  }

  {
    VkPipelineLayoutCreateInfo pipe_layout_info;
    MEMZERO(pipe_layout_info);
    pipe_layout_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_info.pSetLayouts    = &out->set_layouts[0];
    pipe_layout_info.setLayoutCount = out->set_layouts.size;
    VkPushConstantRange push_range;
    push_range.offset     = 0;
    push_range.stageFlags = VK_SHADER_STAGE_ALL;
    push_range.size       = table_info.push_constants_size;
    ;
    if (push_range.size > 0) {
      pipe_layout_info.pPushConstantRanges    = &push_range;
      pipe_layout_info.pushConstantRangeCount = 1;
    }
    VK_ASSERT_OK(
        vkCreatePipelineLayout(dev_ctx->device, &pipe_layout_info, NULL, &out->pipeline_layout));
  }
  return out;
}

VK_Binding_Table *VK_Binding_Table::create(VkDeviceContext *     dev_ctx,
                                           VK_Binding_Signature *signature) {
  VK_Binding_Table *out = new VK_Binding_Table;
  out->signature        = signature;
  ito(signature->set_layouts.size) out->sets.push(dev_ctx->allocate_set(signature->set_layouts[i]));
  ito(out->sets.size) out->set_ids.push(dev_ctx->sets.push(DescriptorSet(out->sets[i])));
  out->dev_ctx = dev_ctx;
  return out;
}
void VK_Binding_Table::bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                                    size_t size) {
  Buffer                 buffer = dev_ctx->buffers.read(buf_id.id);
  VkDescriptorBufferInfo binfo;
  MEMZERO(binfo);
  binfo.buffer = buffer.buffer;
  binfo.offset = offset;
  if (size == 0) size = VK_WHOLE_SIZE;
  binfo.range = size;
  VkWriteDescriptorSet wset;
  MEMZERO(wset);
  wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wset.descriptorCount = 1;
  wset.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  wset.dstArrayElement = 0;
  wset.dstBinding      = binding;
  wset.dstSet          = sets[space];
  wset.pBufferInfo     = &binfo;
  vkUpdateDescriptorSets(dev_ctx->device, 1, &wset, 0, NULL);
}
void VK_Binding_Table::bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) {
  VkDescriptorImageInfo binfo;
  MEMZERO(binfo);
  Sampler sampler   = dev_ctx->samplers.read(sampler_id.id);
  binfo.imageLayout = VK_IMAGE_LAYOUT_MAX_ENUM;
  binfo.imageView   = VK_NULL_HANDLE;
  binfo.sampler     = sampler.sampler;
  VkWriteDescriptorSet wset;
  MEMZERO(wset);
  wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wset.descriptorCount = 1;
  wset.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
  wset.dstArrayElement = 0;
  wset.dstBinding      = binding;
  wset.dstSet          = sets[space];
  wset.pImageInfo      = &binfo;
  vkUpdateDescriptorSets(dev_ctx->device, 1, &wset, 0, NULL);
}
void VK_Binding_Table::bind_UAV_buffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                                       size_t size) {
  Buffer                 buffer = dev_ctx->buffers.read(buf_id.id);
  VkDescriptorBufferInfo binfo;
  MEMZERO(binfo);
  binfo.buffer = buffer.buffer;
  binfo.offset = offset;
  if (size == 0) size = VK_WHOLE_SIZE;
  binfo.range = size;
  VkWriteDescriptorSet wset;
  MEMZERO(wset);
  wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wset.descriptorCount = 1;
  wset.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wset.dstArrayElement = 0;
  wset.dstBinding      = binding;
  wset.dstSet          = sets[space];
  wset.pBufferInfo     = &binfo;
  vkUpdateDescriptorSets(dev_ctx->device, 1, &wset, 0, NULL);
}
void VK_Binding_Table::bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                                    rd::Image_Subresource const &range, rd::Format format) {
  VkDescriptorImageInfo binfo;
  MEMZERO(binfo);
  Image    img = dev_ctx->images.read(image_id.id);
  VkFormat vkformat{};
  if (format == rd::Format::NATIVE)
    vkformat = img.info.format;
  else
    vkformat = to_vk(format);
  u32 num_levels = range.num_levels;
  if (num_levels == -1) num_levels = img.info.mipLevels;
  u32 num_layers = range.num_layers;
  if (num_layers == -1) num_layers = img.info.arrayLayers;
  ID view_id = dev_ctx
                   ->create_image_view(img.id, range.level, num_levels, range.layer,
                                       range.num_layers, vkformat)
                   .id;
  ImageView view    = dev_ctx->image_views.read(view_id);
  binfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  binfo.imageView   = view.view;
  binfo.sampler     = VK_NULL_HANDLE;
  VkWriteDescriptorSet wset;
  MEMZERO(wset);
  wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wset.descriptorCount = 1;
  wset.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  wset.dstArrayElement = index;
  wset.dstBinding      = binding;
  wset.dstSet          = sets[space];
  wset.pImageInfo      = &binfo;
  vkUpdateDescriptorSets(dev_ctx->device, 1, &wset, 0, NULL);
}
void VK_Binding_Table::bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                                        rd::Image_Subresource const &range, rd::Format format) {
  VkDescriptorImageInfo binfo;
  MEMZERO(binfo);
  Image    img = dev_ctx->images.read(image_id.id);
  VkFormat vkformat{};
  if (format == rd::Format::NATIVE)
    vkformat = img.info.format;
  else
    vkformat = to_vk(format);
  ID view_id = dev_ctx
                   ->create_image_view(img.id, range.level, range.num_levels, range.layer,
                                       range.num_layers, vkformat)
                   .id;
  ImageView view    = dev_ctx->image_views.read(view_id);
  binfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  binfo.imageView   = view.view;
  binfo.sampler     = VK_NULL_HANDLE;
  VkWriteDescriptorSet wset;
  MEMZERO(wset);
  wset.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wset.descriptorCount = 1;
  wset.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  wset.dstArrayElement = index;
  wset.dstBinding      = binding;
  wset.dstSet          = sets[space];
  wset.pImageInfo      = &binfo;
  vkUpdateDescriptorSets(dev_ctx->device, 1, &wset, 0, NULL);
}
void VK_Binding_Table::release() {
  // ito(sets.size) dev_ctx->desc_pool.free(sets[i]);
  ito(set_ids.size) dev_ctx->sets.remove(set_ids[i], 3);
  delete this;
}
void VK_Binding_Signature::release(VkDeviceContext *dev_ctx) {
  ito(set_layouts.size) vkDestroyDescriptorSetLayout(dev_ctx->device, set_layouts[i], NULL);
  vkDestroyPipelineLayout(dev_ctx->device, pipeline_layout, NULL);
  delete this;
}
} // namespace
namespace rd {
IDevice *create_vulkan(void *window_handler) { return new VkFactory(window_handler); }
} // namespace rd
