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

// 1 <<  8 = alignment
// 1 << 17 = num of blocks
struct Mem_Chunk : Slot {
  u32                   ref_cnt          = 0;
  VkDeviceMemory        mem              = VK_NULL_HANDLE;
  VkMemoryPropertyFlags prop_flags       = 0;
  static constexpr u32  PAGE_SIZE        = 0x100;
  u32                   size             = 0;
  u32                   cursor           = 0; // points to the next free 256 byte block
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
  void rem_reference() {
    ASSERT_DEBUG(ref_cnt > 0);
    ref_cnt--;
  }
  bool is_referenced() { return ref_cnt != 0; }
  void release(VkDevice device) { vkFreeMemory(device, mem, NULL); }
  bool has_space(u32 req_size) {
    if (ref_cnt == 0) cursor = 0;
    return cursor + ((req_size + PAGE_SIZE - 1) / PAGE_SIZE) < size;
  }
  u32 alloc(u32 alignment, u32 req_size) {
    ASSERT_DEBUG((alignment & (alignment - 1)) == 0); // PoT
    ASSERT_DEBUG(((alignment - 1) & PAGE_SIZE) == 0); // 256 bytes is enough to align this
    u32 offset = cursor;
    cursor += ((req_size + PAGE_SIZE - 1) / PAGE_SIZE);
    ASSERT_DEBUG(cursor < size);
    ref_cnt++;
    return offset * PAGE_SIZE;
  }
};

struct Buffer : public Slot {
  u32                mem_chunk_index;
  u32                mem_offset;
  VkBuffer           buffer;
  VkBufferCreateInfo create_info;
  VkAccessFlags      access_flags;
};

struct Image : public Slot {
  u32                mem_chunk_index;
  u32                mem_offset;
  VkImageLayout      layout;
  VkAccessFlags      access_flags;
  VkImageAspectFlags aspect;
  VkImage            image;
  VkImageCreateInfo  create_info;
};

struct Shader : public Slot {
  VkShaderModule module;
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
    fprintf(stdout, "Resource_Array:");
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

struct Shader_Descriptor {
  u32                          set;
  VkDescriptorSetLayoutBinding layout;
};

struct Vertex_Input {
  uint32_t binding;
  uint32_t offset;
  VkFormat format;
};

struct Graphics_Pipeline_State {
  VkCullModeFlags     cull_mode;
  VkFrontFace         front_face;
  VkPolygonMode       polygon_mode;
  float               line_width;
  bool                enable_depth_test;
  VkCompareOp         cmp_op;
  bool                enable_depth_write;
  float               max_depth;
  VkPrimitiveTopology topology;
  float               depth_bias_const;
  u32                 ps, vs;
  u32                 pass;
  u64                 dummy; // used for hashing to emulate C string
  bool                operator==(const Graphics_Pipeline_State &that) const {
    return memcmp(this, &that, sizeof(*this)) == 0;
  }
  Graphics_Pipeline_State() {
    memset(this, 0, sizeof(*this)); // Important for memhash
    dummy = 0;
  }
};

u64 hash_of(Graphics_Pipeline_State const &state) { return hash_of((char const *)&state); }

struct Graphics_Pipeline_Wrapper : public Slot {
  VkShaderModule                            vs_shader_module;
  VkShaderModule                            ps_shader_module;
  SmallArray<VkDescriptorSetLayout, 4>      set_layouts;
  VkPipelineLayout                          pipeline_layout;
  VkPipeline                                pipeline;
  Hash_Table<string_ref, Shader_Descriptor> resource_slots;

  void release(VkDevice device) {
    vkDestroyShaderModule(device, vs_shader_module, NULL);
    vkDestroyShaderModule(device, ps_shader_module, NULL);
    ito(set_layouts.size) vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    resource_slots.release();
  }

  //  void init(VkDevice                         device,                   //
  //            VkShaderModule                   vs_shader_module,         //
  //            VkShaderModule                   ps_shader_module,         //
  //            VkGraphicsPipelineCreateInfo     pipeline_create_template, //
  //            Pair<string_ref, Vertex_Input> * vertex_inputs,            //
  //            u32                              num_vertex_inputs,        //
  //            VkVertexInputBindingDescription *vertex_bind_descs,        //
  //            u32                              num_vertex_bind_descs,    //
  //            u32                              push_constants_size = 128) {}
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
    Window *wnd = NULL;
    void    release_item(Buffer &buf) {
      vkDestroyBuffer(wnd->device, buf.buffer, NULL);
      wnd->mem_chunks[buf.mem_chunk_index].rem_reference();
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } buffers;
  struct Image_Array : Resource_Array<Image, Image_Array> {
    Window *wnd = NULL;
    void    release_item(Image &img) {
      vkDestroyImage(wnd->device, img.image, NULL);
      wnd->mem_chunks[img.mem_chunk_index].rem_reference();
    }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } images;
  struct Shader_Array : Resource_Array<Shader, Shader_Array> {
    Window *wnd = NULL;
    void release_item(Shader &shader) { vkDestroyShaderModule(wnd->device, shader.module, NULL); }
    void init(Window *wnd) {
      this->wnd = wnd;
      Resource_Array::init();
    }
  } shaders;

  void init_ds() {
    mem_chunks.init();
    buffers.init(this);
    images.init(this);
    shaders.init(this);
  }

  void release() {
    buffers.release();
    images.release();
    shaders.release();
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
    ito(mem_chunks.size) { // look for a suitable memory chunk
      Mem_Chunk &chunk = mem_chunks[i];
      if ((chunk.prop_flags & prop_flags) == prop_flags &&
          (chunk.memory_type_bits & reqs.memoryTypeBits) == reqs.memoryTypeBits) {
        if (chunk.has_space(info.size)) {
          u32 offset = chunk.alloc(reqs.alignment, info.size);

          new_buf.buffer          = buf;
          new_buf.access_flags    = 0;
          new_buf.create_info     = cinfo;
          new_buf.mem_chunk_index = i;
          new_buf.mem_offset      = offset;
          goto finally;
        }
      }
    }
    // if failed create a new one
    {
      Mem_Chunk new_chunk;
      u32       num_pages = 1 << 17;
      if (num_pages * Mem_Chunk::PAGE_SIZE < info.size) {
        num_pages = (info.size + Mem_Chunk::PAGE_SIZE - 1) / Mem_Chunk::PAGE_SIZE;
      }
      new_chunk.init(device, num_pages, find_mem_type(reqs.memoryTypeBits, prop_flags), prop_flags,
                     reqs.memoryTypeBits);

      ASSERT_DEBUG(new_chunk.has_space(info.size));
      u32 offset              = new_chunk.alloc(reqs.alignment, info.size);
      new_buf.buffer          = buf;
      new_buf.access_flags    = 0;
      new_buf.create_info     = cinfo;
      new_buf.mem_chunk_index = (u32)mem_chunks.size;
      new_buf.mem_offset      = offset;
      mem_chunks.push(new_chunk);
    }
  finally:
    if (initial_data != NULL) {
      ASSERT_DEBUG(info.mem_bits & (i32)rd::Memory_Bits::MAPPABLE);
      Mem_Chunk &chunk = mem_chunks[new_buf.mem_chunk_index];
      void *     data  = NULL;
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
    Mem_Chunk &chunk = mem_chunks[buf.mem_chunk_index];
    void *     data  = NULL;
    VK_ASSERT_OK(vkMapMemory(device, chunk.mem, buf.mem_offset, buf.create_info.size, 0, &data));
    return data;
  }

  void unmap_buffer(Resource_ID res_id) {
    ASSERT_DEBUG(res_id.type == (i32)rd::Type::Buffer);
    Buffer &   buf   = buffers[res_id.id];
    Mem_Chunk &chunk = mem_chunks[buf.mem_chunk_index];
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

  Resource_ID alloc_shader(size_t len, u32 *bytecode) {
    Shader sh;
    sh.module = compile_spirv(len, bytecode);
    return {shaders.push(sh), (i32)rd::Type::Shader};
  }

  void release_resource(Resource_ID res_id) {
    if (res_id.type == (i32)rd::Type::Buffer) {
      buffers.remove(res_id.id, 3);
    } else if (res_id.type == (i32)rd::Type::Shader) {
      shaders.remove(res_id.id, 3);
    } else {
      TRAP;
    }
  }

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
      //          physdevice_handles[i], &num_queue_family_properties, queue_family_properties2KHR);
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
    buffers.tick();
    images.tick();
    shaders.tick();
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

struct Shader_Builder {
  Pool<char>          tmp_buf;
  u32                 input_counter   = 0;
  u32                 output_counter  = 0;
  u32                 set_counter     = 0;
  u32                 binding_counter = 0;
  shaderc_shader_kind kind;
  void                init() { tmp_buf = Pool<char>::create(1 << 20); }
  void                release() { tmp_buf.release(); }
  string_ref          get_str() { return string_ref{(char const *)tmp_buf.at(0), tmp_buf.cursor}; }
  void                putf(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    i32 len = vsprintf(tmp_buf.back(), fmt, args);
    va_end(args);
    ASSERT_ALWAYS(len > 0);
    tmp_buf.advance(len);
  }
  void eval(List *l) {
    if (l->child != NULL) {
      eval(l->child);
    } else if (l->cmp_symbol("type")) {
      if (l->next->cmp_symbol("pixel")) {
        kind = shaderc_glsl_fragment_shader;
      } else if (l->next->cmp_symbol("compute")) {
        kind = shaderc_glsl_compute_shader;
      } else if (l->next->cmp_symbol("vertex")) {
        kind = shaderc_glsl_vertex_shader;
      } else {
        TRAP;
      }
    } else if (l->cmp_symbol("build_shader")) {
      putf("#version 450\n");
      putf("#extension GL_EXT_nonuniform_qualifier : require\n");
      List *cur = l->next;
      while (cur != NULL) {
        eval(cur);
        cur = cur->next;
      }
    } else if (l->cmp_symbol("body")) {
      tmp_buf.put(l->get(1)->symbol.ptr, l->get(1)->symbol.len);
    } else if (l->cmp_symbol("input")) {
      putf("layout(location = %i) in %.*s %.*s;\n", input_counter, STRF(l->next->symbol),
           STRF(l->next->next->symbol));
      input_counter += 1;
    } else if (l->cmp_symbol("output")) {
      putf("layout(location = %i) out %.*s %.*s;\n", output_counter, STRF(l->next->symbol),
           STRF(l->next->next->symbol));
      output_counter += 1;
    } else if (l->cmp_symbol("push_constants")) {
      string_ref name = l->next->symbol;
      putf("layout(push_constant) uniform %.*s_t {\n", STRF(name));
      List *cur = l->get(2);
      while (cur != NULL) {
        eval(cur);
        cur = cur->next;
      }
      putf("} %.*s;\n", STRF(name));
    } else if (l->cmp_symbol("set")) {
      binding_counter = 0;
      List *cur       = l->next;
      while (cur != NULL) {
        eval(cur);
        cur = cur->next;
      }
      set_counter += 1;
    } else if (l->cmp_symbol("uniform_buffer")) {
      string_ref name = l->next->symbol;
      putf("layout(set = %i, binding = %i, std140) uniform %.*s_t {\n", set_counter,
           binding_counter, STRF(name));
      List *cur = l->get(2);
      while (cur != NULL) {
        eval(cur);
        cur = cur->next;
      }
      putf("} %.*s;\n", STRF(name));
      binding_counter += 1;
    } else if (l->cmp_symbol("uniform_array")) {
      string_ref type  = l->get(1)->symbol;
      string_ref name  = l->get(2)->symbol;
      string_ref cnt_s = l->get(3)->symbol;
      i32        cnt   = 0;
      ASSERT_ALWAYS(parse_decimal_int(cnt_s.ptr, cnt_s.len, &cnt));
      putf("layout(set = %i, binding = %i) uniform %.*s %.*s [%i];\n", set_counter, binding_counter,
           STRF(type), STRF(name), cnt);
      binding_counter += cnt;
    } else if (l->cmp_symbol("member")) {
      putf("  %.*s %.*s;\n", STRF(l->next->symbol), STRF(l->next->next->symbol));
      input_counter += 1;
    }
  }
};

// typedef void (*Loop_Callback_t)(rd::Pass_Mng *);

enum class Render_Value_t {
  UNKNOWN = 0,
  RESOURCE_ID,
  FLAGS,
  ARRAY,
  SHADER_SOURCE,
};

struct Render_Value {
  struct Array {
    void *ptr;
    u32   size;
  };
  struct Shader_Source {
    shaderc_shader_kind kind;
    string_ref          text;
  };
  union {
    Resource_ID   res_id;
    Array         arr;
    Shader_Source shsrc;
  };
};

struct Renderign_Evaluator final : public IEvaluator {
  Window             wnd;
  Pool<Render_Value> rd_values;
  Pool<>             tmp_values;
  void               init() {
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
  }
  void exit_scope() {
    state->exit_scope();
    tmp_values.exit_scope();
    rd_values.exit_scope();
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
  Value *wrap_shader_source(string_ref text, shaderc_shader_kind kind) {
    Render_Value *rval = rd_values.alloc_zero(1);
    rval->shsrc.kind   = kind;
    rval->shsrc.text   = text;
    Value *new_val     = state->value_storage.alloc_zero(1);
    new_val->type      = (i32)Value::Value_t::ANY;
    new_val->any_type  = (i32)Render_Value_t::SHADER_SOURCE;
    new_val->any       = rval;
    return new_val;
  }
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
      SmallArray<Value *, 2> args;
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
    } else if (l->cmp_symbol("build_shader")) {
      Shader_Builder builder;
      builder.init();
      builder.eval(l);
      string_ref          text = move_cstr(builder.get_str());
      shaderc_shader_kind kind = builder.kind;
      builder.release();
      return wrap_shader_source(text, kind);
    } else if (l->cmp_symbol("compile_shader")) {
      SmallArray<Value *, 4> args;
      args.init();
      defer(args.release());
      eval_args_and_collect(l->next, args);
      ASSERT_EVAL(args.size == 1);
      ASSERT_EVAL(args[0]->type == (i32)Value::Value_t::ANY &&
                  args[0]->any_type == (i32)Render_Value_t::SHADER_SOURCE);
      Render_Value *            val_1    = (Render_Value *)args[0]->any;
      string_ref                text     = val_1->shsrc.text;
      shaderc_shader_kind       kind     = val_1->shsrc.kind;
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
        state->push_error("%.*s\n", STRF(text));
        state->push_error(shaderc_result_get_error_message(result));
        TRAP;
      }
      size_t         len    = shaderc_result_get_length(result);
      u32 *          spv    = (u32 *)shaderc_result_get_bytes(result);
      Resource_ID res_id = wnd.alloc_shader(len, spv);
      return wrap_resource(res_id);
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
