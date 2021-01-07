#include "script.hpp"
#include "utils.hpp"

#include <functional>

#include <SDL.h>
#include <SDL_syswm.h>
#include <atomic>
#include <mutex>
#include <thread>

#include "rendering.hpp"

#define WIN32_LEAN_AND_MEAN
#include <DirectXMath.h>
#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxcapi.h>
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

D3D12_CULL_MODE to_dx(rd::Cull_Mode op) {
  switch (op) {
  case rd::Cull_Mode::BACK: return D3D12_CULL_MODE_BACK;
  case rd::Cull_Mode::FRONT: return D3D12_CULL_MODE_FRONT;
  case rd::Cull_Mode::NONE: return D3D12_CULL_MODE_NONE;
  default: {
    TRAP;
  }
  }
}
D3D12_BLEND_OP to_dx(rd::Blend_OP op) {
  switch (op) {
  case rd::Blend_OP::ADD: return D3D12_BLEND_OP_ADD;
  case rd::Blend_OP::MAX: return D3D12_BLEND_OP_MAX;
  case rd::Blend_OP::MIN: return D3D12_BLEND_OP_MIN;
  case rd::Blend_OP::REVERSE_SUBTRACT: return D3D12_BLEND_OP_REV_SUBTRACT;
  case rd::Blend_OP::SUBTRACT: return D3D12_BLEND_OP_SUBTRACT;
  default: {
    TRAP;
  }
  }
}
D3D12_PRIMITIVE_TOPOLOGY_TYPE to_dx_type(rd::Primitive op) {
  switch (op) {
  case rd::Primitive::LINE_LIST: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
  case rd::Primitive::TRIANGLE_LIST: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  case rd::Primitive::TRIANGLE_STRIP: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  default: {
    TRAP;
  }
  }
}
D3D12_PRIMITIVE_TOPOLOGY to_dx(rd::Primitive op) {
  switch (op) {
  case rd::Primitive::LINE_LIST: return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
  case rd::Primitive::TRIANGLE_LIST: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
  case rd::Primitive::TRIANGLE_STRIP: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
  default: {
    TRAP;
  }
  }
}
D3D12_BLEND to_dx(rd::Blend_Factor factor) {
  switch (factor) {
  // clang-format off
    case rd::Blend_Factor::ZERO                     : return D3D12_BLEND_ZERO;
    case rd::Blend_Factor::ONE                      : return D3D12_BLEND_ONE;
    case rd::Blend_Factor::SRC_COLOR                : return D3D12_BLEND_SRC_COLOR;
    case rd::Blend_Factor::ONE_MINUS_SRC_COLOR      : return D3D12_BLEND_INV_SRC_COLOR;
    case rd::Blend_Factor::DST_COLOR                : return D3D12_BLEND_DEST_COLOR;
    case rd::Blend_Factor::ONE_MINUS_DST_COLOR      : return D3D12_BLEND_INV_DEST_COLOR;
    case rd::Blend_Factor::SRC_ALPHA                : return D3D12_BLEND_SRC_ALPHA;
    case rd::Blend_Factor::ONE_MINUS_SRC_ALPHA      : return D3D12_BLEND_INV_SRC_ALPHA;
    case rd::Blend_Factor::DST_ALPHA                : return D3D12_BLEND_DEST_ALPHA;
    case rd::Blend_Factor::ONE_MINUS_DST_ALPHA      : return D3D12_BLEND_INV_DEST_ALPHA;
    case rd::Blend_Factor::CONSTANT_COLOR           : return D3D12_BLEND_SRC1_COLOR;
    case rd::Blend_Factor::ONE_MINUS_CONSTANT_COLOR : return D3D12_BLEND_INV_SRC1_COLOR;
    case rd::Blend_Factor::CONSTANT_ALPHA           : return D3D12_BLEND_SRC1_ALPHA;
    case rd::Blend_Factor::ONE_MINUS_CONSTANT_ALPHA : return D3D12_BLEND_INV_SRC1_ALPHA;
    // clang-format on
  default: {
    TRAP;
  }
  }
}
DXGI_FORMAT to_dx(rd::Index_t format) {
  // clang-format off
  switch (format) {
  case rd::Index_t::UINT16     : return DXGI_FORMAT_R16_UINT      ;
  case rd::Index_t::UINT32     : return DXGI_FORMAT_R32_UINT      ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}
DXGI_FORMAT to_dx(rd::Format format) {
  // clang-format off
  switch (format) {
  case rd::Format::RGBA8_UNORM     : return DXGI_FORMAT_R8G8B8A8_UNORM      ;
  case rd::Format::RGBA8_SNORM     : return DXGI_FORMAT_R8G8B8A8_SNORM      ;
  case rd::Format::RGBA8_SRGBA     : return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB       ;
  case rd::Format::RGBA8_UINT      : return DXGI_FORMAT_R8G8B8A8_UINT;
  case rd::Format::BGRA8_SRGBA     : return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB       ;
  case rd::Format::RGBA32_FLOAT    : return DXGI_FORMAT_R32G32B32A32_FLOAT ;
  case rd::Format::RGB32_FLOAT     : return DXGI_FORMAT_R32G32B32_FLOAT    ;
  case rd::Format::RG32_FLOAT      : return DXGI_FORMAT_R32G32_FLOAT       ;
  case rd::Format::R32_FLOAT       : return DXGI_FORMAT_R32_FLOAT          ;
  case rd::Format::R16_FLOAT       : return DXGI_FORMAT_R16_FLOAT          ;
  case rd::Format::R16_UNORM       : return DXGI_FORMAT_R16_UNORM;
  case rd::Format::R8_UNORM        : return DXGI_FORMAT_R8_UNORM;
  case rd::Format::D32_FLOAT       : return DXGI_FORMAT_D32_FLOAT          ;
  case rd::Format::R32_UINT        : return DXGI_FORMAT_R32_UINT;
  case rd::Format::R16_UINT        : return DXGI_FORMAT_R16_UINT            ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

rd::Format from_dx(DXGI_FORMAT format) {
  // clang-format off
  switch (format) {
  case DXGI_FORMAT_R8G8B8A8_UNORM     : return rd::Format::RGBA8_UNORM ;
  case DXGI_FORMAT_R8G8B8A8_SNORM     : return rd::Format::RGBA8_SNORM ;
  case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return rd::Format::RGBA8_SRGBA ;
  case DXGI_FORMAT_R8G8B8A8_UINT      : return rd::Format::RGBA8_UINT  ;
  case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return rd::Format::BGRA8_SRGBA ;
  case DXGI_FORMAT_R32G32B32A32_FLOAT : return rd::Format::RGBA32_FLOAT;
  case DXGI_FORMAT_R32G32B32_FLOAT    : return rd::Format::RGB32_FLOAT ;
  case DXGI_FORMAT_R32G32_FLOAT       : return rd::Format::RG32_FLOAT  ;
  case DXGI_FORMAT_R32_FLOAT          : return rd::Format::R32_FLOAT   ;
  case DXGI_FORMAT_R16_FLOAT          : return rd::Format::R16_FLOAT   ;
  case DXGI_FORMAT_R16_UNORM          : return rd::Format::R16_UNORM   ;
  case DXGI_FORMAT_R8_UNORM           : return rd::Format::R8_UNORM    ;
  case DXGI_FORMAT_D32_FLOAT          : return rd::Format::D32_FLOAT   ;
  case DXGI_FORMAT_R32_UINT           : return rd::Format::R32_UINT    ;
  case DXGI_FORMAT_R16_UINT           : return rd::Format::R16_UINT    ;
  default: UNIMPLEMENTED;
  }
  // clang-format on
}

D3D12_RESOURCE_STATES to_dx(rd::Image_Access access) {
  switch (access) {
  case rd::Image_Access::GENERIC: return D3D12_RESOURCE_STATE_COMMON;
  case rd::Image_Access::DEPTH_TARGET: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
  case rd::Image_Access::COLOR_TARGET: return D3D12_RESOURCE_STATE_RENDER_TARGET;
  case rd::Image_Access::SAMPLED: return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  case rd::Image_Access::TRANSFER_DST: return D3D12_RESOURCE_STATE_COPY_DEST;
  case rd::Image_Access::TRANSFER_SRC: return D3D12_RESOURCE_STATE_COPY_SOURCE;
  case rd::Image_Access::UAV: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  default: {
    TRAP;
  }
  }
}
D3D12_RESOURCE_STATES to_dx(rd::Buffer_Access access) {
  switch (access) {
  case rd::Buffer_Access::GENERIC: return D3D12_RESOURCE_STATE_COMMON;
  case rd::Buffer_Access::UNIFORM: return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
  case rd::Buffer_Access::VERTEX_BUFFER: return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
  case rd::Buffer_Access::INDEX_BUFFER: return D3D12_RESOURCE_STATE_INDEX_BUFFER;
  case rd::Buffer_Access::UAV: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  case rd::Buffer_Access::TRANSFER_DST: return D3D12_RESOURCE_STATE_COPY_DEST;
  case rd::Buffer_Access::TRANSFER_SRC: return D3D12_RESOURCE_STATE_COPY_SOURCE;
  case rd::Buffer_Access::HOST_READ: return D3D12_RESOURCE_STATE_GENERIC_READ;
  case rd::Buffer_Access::HOST_WRITE: return D3D12_RESOURCE_STATE_COMMON;
  case rd::Buffer_Access::HOST_READ_WRITE: return D3D12_RESOURCE_STATE_COMMON;

  default: {
    TRAP;
  }
  }
}
D3D12_TEXTURE_ADDRESS_MODE to_dx(rd::Address_Mode mode) {
  switch (mode) {
  case rd::Address_Mode::CLAMP_TO_EDGE: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  case rd::Address_Mode::MIRRORED_REPEAT: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
  case rd::Address_Mode::REPEAT: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
  default: TRAP;
  }
}

// D3D12_RESOURCE_STATES get_default_state(ID3D12Resource *res) {
//  D3D12_HEAP_PROPERTIES hp{};
//  D3D12_HEAP_FLAGS      hf{};
//  DX_ASSERT_OK(res->GetHeapProperties(&hp, &hf));
//  if (hp.Type == D3D12_HEAP_TYPE_READBACK) //
//    return D3D12_RESOURCE_STATE_COPY_DEST;
//  if (hp.Type == D3D12_HEAP_TYPE_UPLOAD) return D3D12_RESOURCE_STATE_GENERIC_READ;
//  if (res->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
//    return D3D12_RESOURCE_STATE_COMMON;
//  return D3D12_RESOURCE_STATE_COMMON;
//}

D3D12_COMPARISON_FUNC to_dx(rd::Cmp mode) {
  switch (mode) {
  case rd::Cmp::EQ: return D3D12_COMPARISON_FUNC_EQUAL;
  case rd::Cmp::GE: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
  case rd::Cmp::GT: return D3D12_COMPARISON_FUNC_GREATER;
  case rd::Cmp::LE: return D3D12_COMPARISON_FUNC_LESS_EQUAL;
  case rd::Cmp::LT: return D3D12_COMPARISON_FUNC_LESS;
  default: TRAP;
  }
}
namespace {
static std::atomic<int> g_thread_counter;
static int              get_thread_id() {
  static thread_local int id = g_thread_counter++;
  return id;
};
} // namespace

// The scope shall not be entered by multiple threads
#define ASSERT_SINGLE_THREAD_SCOPE                                                                 \
  _grab_st_lock();                                                                                 \
  defer(_release_st_lock());

// Just mutex
#define SCOPED_LOCK std::lock_guard<std::mutex> _lock(mutex);

class GPU_Desc_Heap {
  private:
  ComPtr<ID3D12DescriptorHeap> heap;
  Util_Allocator               ual;
  D3D12_DESCRIPTOR_HEAP_TYPE   type;
  SIZE_T                       element_size;
  std::mutex                   mutex;

  public:
  SIZE_T                       get_element_size() { return element_size; }
  D3D12_DESCRIPTOR_HEAP_TYPE   get_type() { return type; }
  ComPtr<ID3D12DescriptorHeap> get_heap() { return heap; }
  GPU_Desc_Heap(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type, u32 size,
                D3D12_DESCRIPTOR_HEAP_FLAGS flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
      : ual(size), type(type) {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type                       = type;
    desc.NumDescriptors             = size;
    desc.Flags                      = flags;
    DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));
    element_size = device->GetDescriptorHandleIncrementSize(type);
  }
  ~GPU_Desc_Heap() {}
  i32 allocate(u32 size) {
    SCOPED_LOCK;
    return ual.alloc(size);
  }
  void free(u32 offset, u32 size) {
    SCOPED_LOCK;
    ual.free(offset, size);
  }
};

enum class Resource_Type : u32 {
  UNKNOWN = 0,
  BUFFER,
  TEXTURE,
  SIGNATURE,
  SHADER,
  SAMPLER,
  COMPUTE_PSO,
  GRAPHICS_PSO,
  RENDER_PASS,
  FRAME_BUFFER,
};

template <typename T> class Resource_Array {
  private:
  AutoArray<T>   items{};
  AutoArray<u32> free_slots{};

  public:
  T  load(ID id) { return items[id.index()]; }
  ID push(T item) {
    if (free_slots.size) {
      u32 index    = free_slots.pop();
      items[index] = item;
      return {index + 1};
    }
    items.push(item);
    return {(u32)items.size};
  }
  void free(ID id) {
    items[id.index()] = {};
    free_slots.push(id.index());
  }
  ~Resource_Array() {}
};

static inline D3D12_HEAP_PROPERTIES get_heap_properties(rd::Memory_Type mem_type) {
  D3D12_HEAP_PROPERTIES prop{};
  prop.Type = D3D12_HEAP_TYPE_DEFAULT;
  if (mem_type == rd::Memory_Type::CPU_WRITE_GPU_READ) {
    prop.CPUPageProperty =
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN; // D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
    prop.Type = D3D12_HEAP_TYPE_UPLOAD;
  } else if (mem_type == rd::Memory_Type::CPU_READ_WRITE) {
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN; // D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    prop.Type            = D3D12_HEAP_TYPE_READBACK;
  } else if (mem_type == rd::Memory_Type::GPU_LOCAL) {
    prop.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN; // D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    prop.Type            = D3D12_HEAP_TYPE_DEFAULT;
  }
  prop.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  prop.CreationNodeMask     = 1;
  prop.VisibleNodeMask      = 1;
  return prop;
}

class DX12Device;

struct DX12Binding_Signature {
  ComPtr<ID3D12RootSignature> root_signature;
  u32                         num_desc_common           = 0;
  u32                         common_desc_heap_ofset    = 0;
  u32                         num_desc_samplers         = 0;
  u32                         samplers_desc_heap_offset = 0;

  i32 common_param_id     = -1;
  i32 samplers_param_id   = -1;
  i32 pc_param_id         = -1;
  u32 num_params          = 0;
  u32 push_constants_size = 0;
  enum { PUSH_CONSTANTS_SPACE = 777 };
  struct ParamDesc {
    InlineArray<D3D12_DESCRIPTOR_RANGE, 0x10> bindings{};
  };
  ParamDesc                     space_descs[0x10]{};
  void                          release() { delete this; }
  static DX12Binding_Signature *create(DX12Device *                         dev_ctx,
                                       rd::Binding_Table_Create_Info const &table_info);
};
class RenderPass;
class FrameBuffer;
struct GraphicsPSOWrapper {
  ID3D12PipelineState *       pso = NULL;
  rd::Graphics_Pipeline_State state{};
  D3D_PRIMITIVE_TOPOLOGY      topology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
};
struct ResourceWrapper {
  ID3D12Resource *      res           = NULL;
  D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATES::D3D12_RESOURCE_STATE_COMMON;
};
class DX12Device : public rd::IDevice {
  static constexpr u32 NUM_BACK_BUFFERS = 3;
  static constexpr u32 MAX_THREADS      = 0x100;

  ComPtr<ID3D12Device2>      device;
  ComPtr<ID3D12CommandQueue> cmd_queue;

  // Swap Chain. Not used when there's no window.
  struct SwapChain_Context {
    bool                    enabled = false;
    ComPtr<IDXGISwapChain4> sc;
    ID                      images[NUM_BACK_BUFFERS]{};
    ComPtr<ID3D12Fence>     fence;
    UINT64                  last_fence_signaled_value = 0;
    HANDLE                  fence_event               = 0;
    HANDLE                  sc_wait_obj               = 0;
    u32                     width                     = 0;
    u32                     height                    = 0;
  } sc{};
  GPU_Desc_Heap *dsv_desc_heap     = NULL;
  GPU_Desc_Heap *rtv_desc_heap     = NULL;
  GPU_Desc_Heap *common_desc_heap  = NULL;
  GPU_Desc_Heap *sampler_desc_heap = NULL;

  struct Frame_Context {
    UINT64                         fence_value = (u64)-1;
    ComPtr<ID3D12CommandAllocator> cmd_allocs[MAX_THREADS];
    u32                            sc_image_id = 0;
  } frame_contexts[NUM_BACK_BUFFERS]{};
  u32  cur_ctx_id = 0;
  bool in_frame   = false;
  // Multi-threading
  std::mutex       mutex;
  std::atomic<i32> single_thread_guard;

  void _grab_st_lock() {
    i32 old = single_thread_guard.exchange((i32)get_thread_id());
    ASSERT_DEBUG(old == -1 || old == (i32)get_thread_id());
  }

  void _release_st_lock() { single_thread_guard.exchange(-1); }

  Resource_Array<ResourceWrapper *>       resource_table;
  Resource_Array<DX12Binding_Signature *> signature_table;
  Resource_Array<IDxcBlob *>              shader_table;
  Resource_Array<ID3D12PipelineState *>   compute_pso_table;
  Resource_Array<GraphicsPSOWrapper *>    graphics_pso_table;
  // Resource_Array<HANDLE>                  events_table;
  Resource_Array<D3D12_SAMPLER_DESC *> sampler_table;
  Resource_Array<RenderPass *>         renderpass_table;
  Resource_Array<FrameBuffer *>        fb_table;

  AutoArray<Pair<Resource_ID, i32>> deferred_release;
  struct Deferred_Descriptor_Range_Release {
    D3D12_DESCRIPTOR_HEAP_TYPE type;
    u32                        offset;
    u32                        size;
    i32                        timeout;
  };
  AutoArray<Deferred_Descriptor_Range_Release> deferred_desc_range_release;

  void deferred_resource_release_iteration();
  void deferred_desc_range_release_iteration();

  ~DX12Device() {}

  // Synchronization
  void sync_transition_barrier(ID3D12Resource *res, D3D12_RESOURCE_STATES StateBefore,
                               D3D12_RESOURCE_STATES StateAfter) {
    ASSERT_SINGLE_THREAD_SCOPE;
    auto                   cmd = alloc_graphics_cmd();
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    bar.Transition.pResource   = res;
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    bar.Transition.StateBefore = StateBefore;
    bar.Transition.StateAfter  = StateAfter;
    cmd->ResourceBarrier(1, &bar);
    DX_ASSERT_OK(cmd->Close());
    cmd_queue->ExecuteCommandLists(1, (ID3D12CommandList **)cmd.GetAddressOf());
    wait_idle();
  }
  void update_sc_images() {
    if (!sc.enabled) return;
    ASSERT_SINGLE_THREAD_SCOPE;
    ito(NUM_BACK_BUFFERS) {
      if (sc.images[i].is_valid()) {
        ResourceWrapper *wr = resource_table.load(sc.images[i]);
        wr->res->Release();
        delete wr;
        resource_table.free(sc.images[i]);
      }
      ID3D12Resource *res = NULL;
      sc.sc->GetBuffer(i, IID_PPV_ARGS(&res));
      sc.images[i] = resource_table.push(new ResourceWrapper{res, D3D12_RESOURCE_STATE_COMMON});
    }
  }
  void wait_for_next_frame() {
    // First use
    if (frame_contexts[cur_ctx_id].fence_value == (u64)-1) return;
    u64 cur_value = sc.fence->GetCompletedValue();
    if (cur_value > frame_contexts[cur_ctx_id].fence_value) //
      return;
    sc.fence->SetEventOnCompletion(frame_contexts[cur_ctx_id].fence_value, sc.fence_event);
    WaitForSingleObject(sc.fence_event, INFINITE);
  }

  public:
  void deferred_release_desc_range(D3D12_DESCRIPTOR_HEAP_TYPE type, u32 offset, u32 size) {
    SCOPED_LOCK;
    deferred_desc_range_release.push({type, offset, size, 6});
  }
  ResourceWrapper *      get_resource(ID id) { return resource_table.load(id); }
  DX12Binding_Signature *get_signature(ID id) { return signature_table.load(id); }
  ID3D12PipelineState *  get_compute_pso(ID id) { return compute_pso_table.load(id); }
  GraphicsPSOWrapper *   get_graphics_pso(ID id) { return graphics_pso_table.load(id); }
  D3D12_SAMPLER_DESC *   get_sampler(ID id) { return sampler_table.load(id); }

  ComPtr<ID3D12Device2>             get_device() { return device; }
  ComPtr<ID3D12GraphicsCommandList> alloc_graphics_cmd() {
    ASSERT_DEBUG(in_frame);
    ComPtr<ID3D12GraphicsCommandList> out;
    DX_ASSERT_OK(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        frame_contexts[cur_ctx_id].cmd_allocs[get_thread_id()].Get(), NULL, IID_PPV_ARGS(&out)));
    return out;
  }
  void bind_desc_heaps(ID3D12GraphicsCommandList *cmd) {
    ID3D12DescriptorHeap *descs[] = {
        common_desc_heap->get_heap().Get(),
        sampler_desc_heap->get_heap().Get(),
    };
    cmd->SetDescriptorHeaps(2, descs);
  }
  GPU_Desc_Heap *get_sampler_desc_heap() { return sampler_desc_heap; }
  GPU_Desc_Heap *get_common_desc_heap() { return common_desc_heap; }
  GPU_Desc_Heap *get_rtv_desc_heap() { return rtv_desc_heap; }
  GPU_Desc_Heap *get_dsv_desc_heap() { return dsv_desc_heap; }
  DX12Device(void *hdl) {
#ifdef DEBUG_BUILD
    {
      ComPtr<ID3D12Debug> debugInterface;
      DX_ASSERT_OK(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
      debugInterface->EnableDebugLayer();
    }
#endif
    single_thread_guard            = false;
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
    DX_ASSERT_OK(D3D12CreateDevice(NULL, featureLevel, IID_PPV_ARGS(&device)));
#ifdef DEBUG_BUILD
    {
      ComPtr<ID3D12InfoQueue> d3dInfoQueue;
      DX_ASSERT_OK(device.As(&d3dInfoQueue));
      d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
      d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
      d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);
      {
        D3D12_MESSAGE_SEVERITY Severities[] = {D3D12_MESSAGE_SEVERITY_INFO};

        D3D12_MESSAGE_ID list[] = {
            D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
            D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
            D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
            D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
            D3D12_MESSAGE_ID_UNMAP_RANGE_NOT_EMPTY,
        };

        D3D12_INFO_QUEUE_FILTER filter = {};
        filter.DenyList.NumSeverities  = ARRAYSIZE(Severities);
        filter.DenyList.pSeverityList  = Severities;
        filter.DenyList.NumIDs         = ARRAYSIZE(list);
        filter.DenyList.pIDList        = list;

        DX_ASSERT_OK(d3dInfoQueue->PushStorageFilter(&filter));
      }
    }
#endif
    dsv_desc_heap     = new GPU_Desc_Heap(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2048,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    rtv_desc_heap     = new GPU_Desc_Heap(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2048,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    sampler_desc_heap = new GPU_Desc_Heap(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048);
    common_desc_heap =
        new GPU_Desc_Heap(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1 << 19);
    DX_ASSERT_OK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&sc.fence)));

    sc.fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    ASSERT_DEBUG(sc.fence_event);
    {
      D3D12_COMMAND_QUEUE_DESC desc = {};
      desc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
      desc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
      desc.NodeMask                 = 1;
      DX_ASSERT_OK(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&cmd_queue)));
    }

    ito(NUM_BACK_BUFFERS) {
      jto(MAX_THREADS) {
        DX_ASSERT_OK(device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame_contexts[i].cmd_allocs[j])));
      }
    }
    if (hdl) {
      sc.enabled = true;
      //{

      //  SIZE_T rtvDescriptorSize =
      //      device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
      //  D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle =
      //      rtv_desc_heap->get_heap()->GetCPUDescriptorHandleForHeapStart();
      //  for (UINT i = 0; i < NUM_BACK_BUFFERS; i++) {
      //    main_rt_desc[i] = rtvHandle;
      //    rtvHandle.ptr += rtvDescriptorSize;
      //  }
      //}

      ComPtr<IDXGIFactory4>   dxgiFactory;
      ComPtr<IDXGISwapChain1> swapChain1;
      DX_ASSERT_OK(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
      DXGI_SWAP_CHAIN_DESC1 sd;
      {
        ZeroMemory(&sd, sizeof(sd));
        sd.BufferCount        = NUM_BACK_BUFFERS;
        sd.Width              = 0;
        sd.Height             = 0;
        sd.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.Flags              = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.SampleDesc.Count   = 1;
        sd.SampleDesc.Quality = 0;
        sd.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode          = DXGI_ALPHA_MODE_UNSPECIFIED;
        sd.Scaling            = DXGI_SCALING_STRETCH;
        sd.Stereo             = FALSE;
      }
      DX_ASSERT_OK(dxgiFactory->CreateSwapChainForHwnd(cmd_queue.Get(), (HWND)hdl, &sd, NULL, NULL,
                                                       &swapChain1));
      DX_ASSERT_OK(swapChain1->QueryInterface(IID_PPV_ARGS(&sc.sc)));
      update_sc_images();
      sc.sc->SetMaximumFrameLatency(NUM_BACK_BUFFERS);
      sc.sc_wait_obj = sc.sc->GetFrameLatencyWaitableObject();
      RECT rect{};
      DX_ASSERT_OK(GetWindowRect((HWND)hdl, &rect));
      sc.width  = rect.right - rect.left;
      sc.height = rect.bottom - rect.top;
      ASSERT_DEBUG(sc.sc_wait_obj);
    }
  }
  Resource_ID create_image(rd::Image_Create_Info info) override {
    SCOPED_LOCK;
    ID3D12Resource *      buf = NULL;
    D3D12_HEAP_PROPERTIES prop{};
    prop.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_FLAGS    flags = D3D12_HEAP_FLAG_NONE;
    D3D12_RESOURCE_DESC desc{};
    if (info.depth == 0) info.depth = 1;
    if (info.width == 0) info.width = 1;
    if (info.height == 0) info.height = 1;
    if (info.depth == 1) {
      if (info.height == 1)
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
      else
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    } else
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Alignment        = 0;
    desc.DepthOrArraySize = info.depth == 1 ? info.layers : info.depth;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;
    if (info.usage_bits & (u32)rd::Image_Usage_Bits::USAGE_UAV)
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (info.usage_bits & (u32)rd::Image_Usage_Bits::USAGE_RT)
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (info.usage_bits & (u32)rd::Image_Usage_Bits::USAGE_DT)
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    desc.Format             = to_dx(info.format);
    desc.Height             = 1;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.MipLevels          = info.levels;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Width              = info.width;
    desc.Height             = info.height;
    auto state              = D3D12_RESOURCE_STATE_COMMON;
    DX_ASSERT_OK(
        device->CreateCommittedResource(&prop, flags, &desc, state, NULL, IID_PPV_ARGS(&buf)));
    return {resource_table.push(new ResourceWrapper{buf, state}), (u32)Resource_Type::TEXTURE};
  }
  Resource_ID create_buffer(rd::Buffer_Create_Info info) override {
    ID3D12Resource *      buf   = NULL;
    D3D12_HEAP_PROPERTIES prop  = get_heap_properties(info.memory_type);
    D3D12_HEAP_FLAGS      flags = D3D12_HEAP_FLAG_NONE;
    D3D12_RESOURCE_DESC   desc{};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment        = 0;
    desc.DepthOrArraySize = 1;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;
    if (info.usage_bits & (u32)rd::Buffer_Usage_Bits::USAGE_UAV)
      desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.Height             = 1;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.MipLevels          = 1;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Width              = info.size;
    if (info.usage_bits & (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER)
      desc.Width = ((desc.Width + 0xffU) & ~0xffu);

    auto state = D3D12_RESOURCE_STATE_COMMON;
    if (prop.Type == D3D12_HEAP_TYPE_READBACK) state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (prop.Type == D3D12_HEAP_TYPE_UPLOAD) state = D3D12_RESOURCE_STATE_GENERIC_READ;

    {
      SCOPED_LOCK;
      DX_ASSERT_OK(
          device->CreateCommittedResource(&prop, flags, &desc, state, NULL, IID_PPV_ARGS(&buf)));
      // ASSERT_DEBUG(get_default_state(buf) == state);
    }
    /*if (state != D3D12_RESOURCE_STATE_COMMON) {
      sync_transition_barrier(buf, state, D3D12_RESOURCE_STATE_COMMON);
    }*/
    return {resource_table.push(new ResourceWrapper{buf, state}), (u32)Resource_Type::BUFFER};
  }
  Resource_ID create_shader(rd::Stage_t type, string_ref text,
                            Pair<string_ref, string_ref> *defines, size_t num_defines) override {
    SCOPED_LOCK;
    static ComPtr<IDxcLibrary>  library;
    static ComPtr<IDxcCompiler> compiler;
    static int                  _init = [] {
      DX_ASSERT_OK(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library)));
      DX_ASSERT_OK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));
      return 0;
    }();
    ComPtr<IDxcBlobEncoding> blob;

    LPCWSTR profile = NULL;
    if (type == rd::Stage_t::VERTEX)
      profile = L"vs_6_2";
    else if (type == rd::Stage_t::PIXEL)
      profile = L"ps_6_2";
    else if (type == rd::Stage_t::COMPUTE)
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
    WCHAR const *               options[] = {L"-Wignored-attributes", L"UNUSED"};
    ComPtr<IDxcOperationResult> result;
    // Do a little preprocessing
    // allocated 1 byte but really the rest of memory
    char *tmp_body = (char *)tl_alloc_tmp(1);
    sprintf(tmp_body, "%s%.*s", R"(
#define DX12_PUSH_CONSTANTS_REGISTER register(b0, space777)
#define u32 uint
#define i32 int
#define f32 float
#define f64 double
#define float2_splat(x)  float2(x, x)
#define float3_splat(x)  float3(x, x, x)
#define float4_splat(x)  float4(x, x, x, x)
      )",
            STRF(text));
    DX_ASSERT_OK(
        library->CreateBlobWithEncodingFromPinned(tmp_body, (u32)strlen(tmp_body), 0, &blob));

    HRESULT hr = compiler->Compile(blob.Get(),                             // pSource
                                   L"shader.hlsl",                         // pSourceName
                                   L"main",                                // pEntryPoint
                                   profile,                                // pTargetProfile
                                   options, (u32)ARRAYSIZE(options) - 1,   // pArguments, argCount
                                   dxc_defines.ptr, (u32)dxc_defines.size, // pDefines, defineCount
                                   NULL,                                   // pIncludeHandler
                                   &result);                               // ppResult
    if (SUCCEEDED(hr)) result->GetStatus(&hr);
    if (FAILED(hr)) {
      if (result) {
        ComPtr<IDxcBlobEncoding> errorsBlob;
        hr = result->GetErrorBuffer(&errorsBlob);
        if (SUCCEEDED(hr) && errorsBlob) {
          fprintf(stdout, "Compilation failed with errors:\n%s\n",
                  (const char *)errorsBlob->GetBufferPointer());
        }
      }
      TRAP;
    } else {
      IDxcBlob *bytecode;
      result->GetResult(&bytecode);
      return {shader_table.push(bytecode), (u32)Resource_Type::SHADER};
    }
  }
  u32         get_num_swapchain_images() override { return NUM_BACK_BUFFERS; }
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) override {
    SCOPED_LOCK;
    // u32                         id = sampler_desc_heap->allocate(1);
    D3D12_SAMPLER_DESC desc{};
    desc.AddressU       = to_dx(info.address_mode_u);
    desc.AddressV       = to_dx(info.address_mode_v);
    desc.AddressW       = to_dx(info.address_mode_w);
    desc.ComparisonFunc = to_dx(info.cmp_op);
    if (info.min_filter == rd::Filter::LINEAR) {
      if (info.mag_filter == rd::Filter::LINEAR) {
        desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
      } else if (info.mag_filter == rd::Filter::NEAREST) {
        desc.Filter = D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
      } else {
        TRAP;
      }
    } else if (info.min_filter == rd::Filter::NEAREST) {
      if (info.mag_filter == rd::Filter::LINEAR) {
        desc.Filter = D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR;
      } else if (info.mag_filter == rd::Filter::NEAREST) {
        desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
      } else {
        TRAP;
      }
    } else {
      TRAP;
    }
    desc.MaxAnisotropy = (u32)info.max_anisotropy;
    desc.MaxLOD        = info.max_lod;
    desc.MinLOD        = info.min_lod;
    desc.MipLODBias    = info.mip_lod_bias;

    /* device->CreateSampler(&desc,
                           {sampler_desc_heap->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr
       + sampler_desc_heap->get_element_size() * id});*/
    return {sampler_table.push(new D3D12_SAMPLER_DESC(desc)), (u32)Resource_Type::SAMPLER};
  } // namespace
  void release_resource(Resource_ID id) override {
    SCOPED_LOCK;
    deferred_release.push({id, 6});
    // if (id.type == (u32)Resource_Type::BUFFER) {
    //  // get_resource(id.id)->Release();
    //
    //} else {
    //  TRAP;
    //}
  }
  Resource_ID create_event() override {
    /* ComPtr<ID3D12Fence> fence;
     {
       SCOPED_LOCK;
       DX_ASSERT_OK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
     }
     HANDLE event = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
     defer(CloseHandle(event));
     DX_ASSERT_OK(fence->SetEventOnCompletion(1, event));
     {
       SCOPED_LOCK;
       cmd_queue->Signal(fence.Get(), 1);
     }
     WaitForSingleObject(event, INFINITE);*/
    TRAP;
  }
  Resource_ID create_timestamp() override {
    SCOPED_LOCK;
    TRAP;
  }
  Resource_ID get_swapchain_image() override {
    return {sc.images[frame_contexts[cur_ctx_id].sc_image_id], (u32)Resource_Type::TEXTURE};
  }
  rd::Image2D_Info get_swapchain_image_info() override {
    SCOPED_LOCK;
    rd::Image2D_Info info{};
    ID3D12Resource * res  = get_resource(sc.images[frame_contexts[cur_ctx_id].sc_image_id])->res;
    auto             desc = res->GetDesc();
    info.height           = desc.Height;
    info.layers           = desc.DepthOrArraySize;
    info.levels           = desc.MipLevels;
    info.width            = desc.Width;
    info.format           = from_dx(desc.Format);
    return info;
  }
  rd::Image_Info get_image_info(Resource_ID res_id) override {
    SCOPED_LOCK;
    rd::Image_Info  info{};
    ID3D12Resource *res  = get_resource(res_id.id)->res;
    auto            desc = res->GetDesc();
    info.depth           = desc.DepthOrArraySize;
    info.height          = desc.Height;
    info.layers          = desc.DepthOrArraySize;
    info.levels          = desc.MipLevels;
    info.width           = desc.Width;
    info.format          = from_dx(desc.Format);
    info.is_depth        = rd::is_depth_format(info.format);
    return info;
  }
  void *map_buffer(Resource_ID id) override {
    // SCOPED_LOCK;
    ID3D12Resource *res = get_resource(id.id)->res;
    D3D12_RANGE     rr{};
    rr.Begin                 = 0;
    rr.End                   = 0;
    D3D12_RESOURCE_DESC desc = res->GetDesc();
    ASSERT_DEBUG(desc.Format = DXGI_FORMAT_R32_TYPELESS);
    D3D12_HEAP_PROPERTIES prop{};
    D3D12_HEAP_FLAGS      flags{};
    DX_ASSERT_OK(res->GetHeapProperties(&prop, &flags));
    if (prop.Type == D3D12_HEAP_TYPE_READBACK) {
      rr.End = desc.Width;
    }
    void *data = NULL;
    DX_ASSERT_OK(res->Map(0, &rr, &data));
    return data;
  }
  void unmap_buffer(Resource_ID id) override {
    // SCOPED_LOCK;
    ID3D12Resource *res = get_resource(id.id)->res;
    D3D12_RANGE     rr{};
    rr.Begin                 = 0;
    rr.End                   = 0;
    D3D12_RESOURCE_DESC desc = res->GetDesc();
    ASSERT_DEBUG(desc.Format = DXGI_FORMAT_R32_TYPELESS);
    rr.End = desc.Width;
    res->Unmap(0, &rr);
  }
  Resource_ID create_render_pass(rd::Render_Pass_Create_Info const &info) override;
  Resource_ID create_frame_buffer(Resource_ID                         render_pass,
                                  rd::Frame_Buffer_Create_Info const &info) override;
  Resource_ID create_compute_pso(Resource_ID signature, Resource_ID cs) override {
    SCOPED_LOCK;
    IDxcBlob *                        bytecode = shader_table.load(cs.id);
    DX12Binding_Signature *           sig      = signature_table.load(signature.id);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.CS.pShaderBytecode  = bytecode->GetBufferPointer();
    desc.CS.BytecodeLength   = bytecode->GetBufferSize();
    desc.pRootSignature      = sig->root_signature.Get();
    desc.Flags               = D3D12_PIPELINE_STATE_FLAG_NONE;
    ID3D12PipelineState *pso = NULL;
    DX_ASSERT_OK(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));
    return {compute_pso_table.push(pso), (u32)Resource_Type::COMPUTE_PSO};
  }
  Resource_ID create_graphics_pso(Resource_ID signature, Resource_ID render_pass,
                                  rd::Graphics_Pipeline_State const &) override;
  Resource_ID create_signature(rd::Binding_Table_Create_Info const &info) override {
    return {signature_table.push(DX12Binding_Signature::create(this, info)),
            (u32)Resource_Type::SIGNATURE};
  }
  rd::IBinding_Table *create_binding_table(Resource_ID signature) override;
  rd::ICtx *          start_render_pass(Resource_ID render_pass, Resource_ID frame_buffer) override;
  void                end_render_pass(rd::ICtx *ctx) override;
  rd::ICtx *          start_compute_pass() override;
  void                end_compute_pass(rd::ICtx *ctx) override;
  bool                get_timestamp_state(Resource_ID) override {
    SCOPED_LOCK;
    TRAP;
  }
  double get_timestamp_ms(Resource_ID t0, Resource_ID t1) override {
    SCOPED_LOCK;
    TRAP;
  }
  void wait_idle() override {
    ComPtr<ID3D12Fence> fence;
    {
      SCOPED_LOCK;
      DX_ASSERT_OK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }
    HANDLE event = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
    defer(CloseHandle(event));
    DX_ASSERT_OK(fence->SetEventOnCompletion(1, event));
    {
      SCOPED_LOCK;
      cmd_queue->Signal(fence.Get(), 1);
    }
    WaitForSingleObject(event, INFINITE);
  }
  bool get_event_state(Resource_ID id) override {
    SCOPED_LOCK;
    TRAP;
  }
  rd::Impl_t getImplType() override { return rd::Impl_t::DX12; }
  void       release() override {
    ASSERT_SINGLE_THREAD_SCOPE; // not a reentrant scope.
    wait_idle();
    ito(0x10) {
      deferred_desc_range_release_iteration();
      deferred_resource_release_iteration();
    }
    if (common_desc_heap) delete common_desc_heap;
    if (sampler_desc_heap) delete sampler_desc_heap;
    if (rtv_desc_heap) delete rtv_desc_heap;
    if (dsv_desc_heap) delete dsv_desc_heap;
    delete this;
  }
  void start_frame() override {
    ASSERT_SINGLE_THREAD_SCOPE; // not a reentrant scope.
    in_frame = true;
    // Handle buffer resizal
    if (sc.enabled) {
      DXGI_SWAP_CHAIN_DESC desc{};
      DX_ASSERT_OK(sc.sc->GetDesc(&desc));
      RECT rect{};
      DX_ASSERT_OK(GetClientRect(desc.OutputWindow, &rect));
      u32 new_width  = rect.right - rect.left;
      u32 new_height = rect.bottom - rect.top;
      if (new_width != sc.width || new_height != sc.height) {
        wait_idle();
        ito(NUM_BACK_BUFFERS) {
          if (sc.images[i].is_valid()) {
            auto wr = resource_table.load(sc.images[i]);
            wr->res->Release();
            delete wr;
            resource_table.free(sc.images[i]);
          }
          sc.images[i] = {};
        }
        DX_ASSERT_OK(sc.sc->ResizeBuffers(NUM_BACK_BUFFERS, new_width, new_height,
                                          DXGI_FORMAT_UNKNOWN,
                                          DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT));
        sc.width  = new_width;
        sc.height = new_height;
        update_sc_images();
      }
    }

    // Now wait to make sure the frame context is free to use
    wait_for_next_frame();

    // Deferred release iteration.
    deferred_desc_range_release_iteration();
    deferred_resource_release_iteration();
    // fprintf(stdout, "%i\n", cur_ctx_id);
    ito(MAX_THREADS) frame_contexts[cur_ctx_id].cmd_allocs[i]->Reset();
    // Prepare the swap chain image for consumption.
    if (sc.enabled) {
      // Switch to the new swap chain image
      frame_contexts[cur_ctx_id].sc_image_id = sc.sc->GetCurrentBackBufferIndex();
      /*auto                   cmd             = alloc_graphics_cmd();
      D3D12_RESOURCE_BARRIER barrier         = {};
      barrier.Type                           = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Flags                          = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      barrier.Transition.pResource =
          get_resource(sc.images[frame_contexts[cur_ctx_id].sc_image_id]);
      barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
      barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
      cmd->ResourceBarrier(1, &barrier);
      DX_ASSERT_OK(cmd->Close());
      cmd_queue->ExecuteCommandLists(1, (ID3D12CommandList **)cmd.GetAddressOf());*/
    }
  }
  void end_frame() override {
    in_frame = false;
    ASSERT_SINGLE_THREAD_SCOPE; // not a reentrant scope.
    // Prepare the swapchain image for presentation.
    if (sc.enabled) {
      /* auto cmd = alloc_graphics_cmd();
       D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource =
            get_resource(sc.images[frame_contexts[cur_ctx_id].sc_image_id]);
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        cmd->ResourceBarrier(1, &barrier);
        DX_ASSERT_OK(cmd->Close());
        cmd_queue->ExecuteCommandLists(1, (ID3D12CommandList **)cmd.GetAddressOf());*/
      sc.sc->Present(0, 0);
    }
    u64 fence_value                        = sc.last_fence_signaled_value + 1;
    frame_contexts[cur_ctx_id].fence_value = fence_value;
    sc.last_fence_signaled_value           = fence_value;
    cmd_queue->Signal(sc.fence.Get(), frame_contexts[cur_ctx_id].fence_value);
    cur_ctx_id = (cur_ctx_id + 1) % NUM_BACK_BUFFERS;
  }
};

D3D12_DESCRIPTOR_RANGE_TYPE to_dx(rd::Binding_t type) {
  switch (type) {
  case rd::Binding_t::READ_ONLY_BUFFER: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  case rd::Binding_t::SAMPLER: return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
  case rd::Binding_t::TEXTURE: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  case rd::Binding_t::UAV_BUFFER: return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  case rd::Binding_t::UAV_TEXTURE: return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  case rd::Binding_t::UNIFORM_BUFFER: return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
  default: TRAP;
  }
}

DX12Binding_Signature *
DX12Binding_Signature::create(DX12Device *                         dev_ctx,
                              rd::Binding_Table_Create_Info const &table_info) {
  DX12Binding_Signature *out                       = new DX12Binding_Signature;
  out->push_constants_size                         = table_info.push_constants_size;
  auto                                      device = dev_ctx->get_device();
  D3D12_ROOT_PARAMETER                      params[3]{};
  InlineArray<D3D12_DESCRIPTOR_RANGE, 0x40> ranges{};
  ito(table_info.spaces.size) {
    auto const &info = table_info.spaces[i];
    jto(info.bindings.size) {
      ASSERT_DEBUG(info.bindings[j].num_array_elems > 0);
      out->space_descs[i].bindings.size = MAX(j + 1, out->space_descs[i].bindings.size);
    }
  }
  {
    D3D12_DESCRIPTOR_RANGE *range_ptr  = ranges.elems + ranges.size;
    u32                     num_ranges = 0;
    ito(table_info.spaces.size) {
      auto const &info = table_info.spaces[i];
      // out->common_parameter_heap_offsets.push(out->num_desc_common);
      jto(info.bindings.size) {
        auto const &b = info.bindings[j];
        // Common descriptors only
        if (b.type == rd::Binding_t::SAMPLER) continue;

        D3D12_DESCRIPTOR_RANGE desc_range            = {};
        desc_range.RangeType                         = to_dx(b.type);
        desc_range.NumDescriptors                    = b.num_array_elems;
        desc_range.BaseShaderRegister                = j;
        desc_range.RegisterSpace                     = i;
        desc_range.OffsetInDescriptorsFromTableStart = out->num_desc_common;
        out->num_desc_common += b.num_array_elems;
        out->space_descs[i].bindings[j] = desc_range;
        ranges.push(desc_range);
        num_ranges++;
      }
    }
    if (num_ranges) {
      out->common_param_id                      = out->num_params;
      D3D12_ROOT_PARAMETER param                = {};
      param.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      param.DescriptorTable.NumDescriptorRanges = num_ranges;
      param.DescriptorTable.pDescriptorRanges   = range_ptr;
      param.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
      params[out->num_params++]                 = param;
    }
  }
  {
    D3D12_DESCRIPTOR_RANGE *range_ptr  = ranges.elems + ranges.size;
    u32                     num_ranges = 0;
    ito(table_info.spaces.size) {
      auto const &info = table_info.spaces[i];
      jto(info.bindings.size) {
        auto const &b = info.bindings[j];
        // Sampler descriptors only
        if (b.type != rd::Binding_t::SAMPLER) continue;

        D3D12_DESCRIPTOR_RANGE desc_range            = {};
        desc_range.RangeType                         = to_dx(b.type);
        desc_range.NumDescriptors                    = b.num_array_elems;
        desc_range.BaseShaderRegister                = j;
        desc_range.RegisterSpace                     = i;
        desc_range.OffsetInDescriptorsFromTableStart = out->num_desc_samplers;
        out->num_desc_samplers += b.num_array_elems;
        out->space_descs[i].bindings[j] = desc_range;
        ranges.push(desc_range);
        num_ranges++;
      }
    }
    if (num_ranges) {
      out->samplers_param_id                    = out->num_params;
      D3D12_ROOT_PARAMETER param                = {};
      param.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      param.DescriptorTable.NumDescriptorRanges = num_ranges;
      param.DescriptorTable.pDescriptorRanges   = range_ptr;
      param.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
      params[out->num_params++]                 = param;
    }
  }
  if (table_info.push_constants_size > 0) {
    out->pc_param_id = out->num_params;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants.ShaderRegister = 0;
    param.Constants.RegisterSpace  = PUSH_CONSTANTS_SPACE;
    param.Constants.Num32BitValues = table_info.push_constants_size / 4;
    param.ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;
    params[out->num_params++]      = param;
  }
  if (out->num_params) {
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters             = out->num_params;
    desc.pParameters               = &params[0];
    desc.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_NONE |
                 D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob *blob = NULL;
    DX_ASSERT_OK(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL));
    DX_ASSERT_OK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                             IID_PPV_ARGS(&out->root_signature)));
    blob->Release();
  }
  return out;
}
class DX12Binding_Table : public rd::IBinding_Table {
  DX12Device *           dev_ctx   = NULL;
  DX12Binding_Signature *signature = NULL;

  ComPtr<ID3D12DescriptorHeap> cpu_common_heap;
  ComPtr<ID3D12DescriptorHeap> cpu_sampler_heap;
  size_t                       common_heap_offset  = 0;
  size_t                       sampler_heap_offset = 0;
  u8                           push_constants_data[128];
  bool                         is_bindings_dirty       = false;
  bool                         is_push_constants_dirty = false;
  bool                         is_bound                = false;

  public:
  static DX12Binding_Table *create(DX12Device *dev_ctx, DX12Binding_Signature *signature) {
    DX12Binding_Table *out = new DX12Binding_Table;
    out->dev_ctx           = dev_ctx;
    out->signature         = signature;
    auto device            = dev_ctx->get_device();

    {
      D3D12_DESCRIPTOR_HEAP_DESC desc = {};
      desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
      desc.NumDescriptors             = signature->num_desc_samplers;
      desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&out->cpu_sampler_heap)));
    }
    {
      D3D12_DESCRIPTOR_HEAP_DESC desc = {};
      desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      desc.NumDescriptors             = signature->num_desc_common;
      desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&out->cpu_common_heap)));
    }
    out->common_heap_offset = dev_ctx->get_common_desc_heap()->allocate(signature->num_desc_common);
    out->sampler_heap_offset =
        dev_ctx->get_sampler_desc_heap()->allocate(signature->num_desc_samplers);
    return out;
  }
  void bind(ComPtr<ID3D12GraphicsCommandList> cmd, rd::Pass_t pass_type) {
    if (pass_type == rd::Pass_t::COMPUTE)
      cmd->SetComputeRootSignature(signature->root_signature.Get());
    else
      cmd->SetGraphicsRootSignature(signature->root_signature.Get());
    if (signature->common_param_id >= 0) {
      if (is_bindings_dirty) {
        dev_ctx->get_device()->CopyDescriptorsSimple(
            signature->num_desc_common,
            {dev_ctx->get_common_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr +
             dev_ctx->get_common_desc_heap()->get_element_size() * common_heap_offset},
            cpu_common_heap->GetCPUDescriptorHandleForHeapStart(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      }
      D3D12_GPU_DESCRIPTOR_HANDLE handle = {
          dev_ctx->get_common_desc_heap()->get_heap()->GetGPUDescriptorHandleForHeapStart().ptr +
          dev_ctx->get_common_desc_heap()->get_element_size() * common_heap_offset};
      if (pass_type == rd::Pass_t::COMPUTE)
        cmd->SetComputeRootDescriptorTable(signature->common_param_id, handle);
      else
        cmd->SetGraphicsRootDescriptorTable(signature->common_param_id, handle);
    }
    if (signature->samplers_param_id >= 0) {
      if (is_bindings_dirty) {
        auto heap_start =
            dev_ctx->get_sampler_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr;
        auto stride = dev_ctx->get_sampler_desc_heap()->get_element_size();
        dev_ctx->get_device()->CopyDescriptorsSimple(
            signature->num_desc_samplers, {heap_start + stride * sampler_heap_offset},
            cpu_sampler_heap->GetCPUDescriptorHandleForHeapStart(),
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
      }
      D3D12_GPU_DESCRIPTOR_HANDLE handle = {
          dev_ctx->get_sampler_desc_heap()->get_heap()->GetGPUDescriptorHandleForHeapStart().ptr +
          dev_ctx->get_sampler_desc_heap()->get_element_size() * sampler_heap_offset};
      if (pass_type == rd::Pass_t::COMPUTE)
        cmd->SetComputeRootDescriptorTable(signature->samplers_param_id, handle);
      else
        cmd->SetGraphicsRootDescriptorTable(signature->samplers_param_id, handle);
    }
    is_bound          = true;
    is_bindings_dirty = false;
  }
  void unbind() { is_bound = false; }
  void flush_push_constants(ComPtr<ID3D12GraphicsCommandList> cmd, rd::Pass_t pass_type) {
    ASSERT_DEBUG(is_bound);
    if (!is_push_constants_dirty) return;
    if (signature->push_constants_size) {
      if (pass_type == rd::Pass_t::COMPUTE)
        cmd->SetComputeRoot32BitConstants(
            signature->pc_param_id, signature->push_constants_size / 4, push_constants_data, 0);
      else
        cmd->SetGraphicsRoot32BitConstants(
            signature->pc_param_id, signature->push_constants_size / 4, push_constants_data, 0);
    }
    is_push_constants_dirty = false;
  }
  void push_constants(void const *data, size_t offset, size_t size) override {
    is_push_constants_dirty = true;
    memcpy(push_constants_data + offset, data, size);
  }
  // u32  get_push_constants_parameter_id() { return num_params_total - 1; }
  // u32  get_push_constants_size() { return push_constants_size; }
  void bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                    size_t size) override {
    ID3D12Resource *res = dev_ctx->get_resource(buf_id.id)->res;
    u32             binding_offset =
        signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart;
    D3D12_CONSTANT_BUFFER_VIEW_DESC desc{};
    ASSERT_DEBUG((offset & 0xffu) == 0);
    desc.BufferLocation = res->GetGPUVirtualAddress() + offset;
    desc.SizeInBytes    = (size + 0xffU) & ~0xffu;
    dev_ctx->get_device()->CreateConstantBufferView(
        &desc, {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
                dev_ctx->get_common_desc_heap()->get_element_size() * ((u64)binding_offset)});
    is_bindings_dirty = true;
  }
  void bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) override {
    D3D12_SAMPLER_DESC *desc = dev_ctx->get_sampler(sampler_id.id);
    u32 offset = signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart;
    dev_ctx->get_device()->CreateSampler(
        desc, {cpu_sampler_heap->GetCPUDescriptorHandleForHeapStart().ptr +
               dev_ctx->get_sampler_desc_heap()->get_element_size() * ((u64)offset)});
    is_bindings_dirty = true;
  }
  void bind_UAV_buffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                       size_t size) override {
    ID3D12Resource *res = dev_ctx->get_resource(buf_id.id)->res;
#ifdef DEBUG_BUILD
    {
      D3D12_RESOURCE_DESC desc = res->GetDesc();
      ASSERT_DEBUG(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      ASSERT_DEBUG((offset & 0x3) == 0);
      ASSERT_DEBUG((size & 0x3) == 0);
    }
#endif
    if (size == 0) {
      size = res->GetDesc().Width;
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
    desc.Buffer.CounterOffsetInBytes = 0;
    desc.Buffer.FirstElement         = (u64)offset / 4;
    desc.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_RAW;
    desc.Buffer.NumElements          = (u32)size / 4;
    desc.Buffer.StructureByteStride  = 0;
    desc.Format                      = DXGI_FORMAT_R32_TYPELESS;
    u32 binding_offset =
        signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart;
    dev_ctx->get_device()->CreateUnorderedAccessView(
        res, NULL, &desc,
        {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
         dev_ctx->get_sampler_desc_heap()->get_element_size() * ((u64)binding_offset)});
    is_bindings_dirty = true;
  }
  void bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                    rd::Image_Subresource const &_range, rd::Format format) override {
    ID3D12Resource *res = dev_ctx->get_resource(image_id.id)->res;
    u32             binding_offset =
        signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart + index;
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    auto                            res_desc = res->GetDesc();
    rd::Image_Subresource           range    = _range;
    if (range.num_layers == -1) range.num_layers = res_desc.DepthOrArraySize;
    if (range.num_levels == -1) range.num_levels = res_desc.MipLevels;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (format == rd::Format::NATIVE)
      desc.Format = res_desc.Format;
    else
      desc.Format = to_dx(format);
    if (res_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
      if (res_desc.DepthOrArraySize == 1) {
        desc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE1D;
        desc.Texture1D.MipLevels       = range.num_levels;
        desc.Texture1D.MostDetailedMip = range.level;
        ASSERT_DEBUG(range.num_layers == 1);
        ASSERT_DEBUG(range.layer == 0);
      } else {
        desc.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
        desc.Texture1DArray.MipLevels       = range.num_levels;
        desc.Texture1DArray.MipLevels       = range.num_levels;
        desc.Texture1DArray.ArraySize       = range.num_layers;
        desc.Texture1DArray.FirstArraySlice = range.layer;
        desc.Texture1DArray.MostDetailedMip = range.level;
      }
    } else if (res_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
      if (res_desc.DepthOrArraySize == 1) {
        desc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
        desc.Texture2D.MipLevels       = range.num_levels;
        desc.Texture2D.MostDetailedMip = range.level;
        ASSERT_DEBUG(range.num_layers == 1);
        ASSERT_DEBUG(range.layer == 0);
      } else {
        desc.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        desc.Texture2DArray.MipLevels       = range.num_levels;
        desc.Texture2DArray.MipLevels       = range.num_levels;
        desc.Texture2DArray.ArraySize       = range.num_layers;
        desc.Texture2DArray.FirstArraySlice = range.layer;
        desc.Texture2DArray.MostDetailedMip = range.level;
      }
    } else if (res_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
      desc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE3D;
      desc.Texture3D.MipLevels       = range.num_levels;
      desc.Texture3D.MostDetailedMip = range.level;
      ASSERT_DEBUG(range.num_layers == 1);
      ASSERT_DEBUG(range.layer == 0);
    }

    dev_ctx->get_device()->CreateShaderResourceView(
        res, &desc,
        {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
         dev_ctx->get_common_desc_heap()->get_element_size() * ((u64)binding_offset)});
    is_bindings_dirty = true;
  }
  void bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                        rd::Image_Subresource const &_range, rd::Format format) override {
    ID3D12Resource *res = dev_ctx->get_resource(image_id.id)->res;
#ifdef DEBUG_BUILD
    {
      D3D12_RESOURCE_DESC desc = res->GetDesc();
      ASSERT_DEBUG(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
#endif
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    auto                             res_desc = res->GetDesc();
    rd::Image_Subresource            range    = _range;
    if (range.num_layers == -1) range.num_layers = res_desc.DepthOrArraySize;
    if (range.num_levels == -1) range.num_levels = res_desc.MipLevels;
    // UAVs can only bind one mip level at a time
    ASSERT_DEBUG(range.num_levels == 1);
    if (res_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
      if (res_desc.DepthOrArraySize == 1) {
        desc.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE1D;
        desc.Texture1D.MipSlice = range.level;
        ASSERT_DEBUG(range.num_layers == 1);
        ASSERT_DEBUG(range.layer == 0);
      } else {
        desc.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
        desc.Texture1DArray.MipSlice        = range.level;
        desc.Texture1DArray.ArraySize       = range.num_layers;
        desc.Texture1DArray.FirstArraySlice = range.layer;
      }
    } else if (res_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
      if (res_desc.DepthOrArraySize == 1) {
        desc.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
        desc.Texture2D.MipSlice = range.level;
        ASSERT_DEBUG(range.num_layers == 1);
        ASSERT_DEBUG(range.layer == 0);
      } else {
        desc.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        desc.Texture2DArray.ArraySize       = range.num_layers;
        desc.Texture2DArray.FirstArraySlice = range.layer;
        desc.Texture2DArray.MipSlice        = range.level;
      }
    } else if (res_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
      desc.ViewDimension         = D3D12_UAV_DIMENSION_TEXTURE3D;
      desc.Texture3D.MipSlice    = range.level;
      desc.Texture3D.FirstWSlice = 0;
      desc.Texture3D.WSize       = res_desc.DepthOrArraySize;
    }

    if (format == rd::Format::NATIVE)
      desc.Format = res_desc.Format;
    else
      desc.Format = to_dx(format);
    u32 binding_offset =
        signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart + index;
    dev_ctx->get_device()->CreateUnorderedAccessView(
        res, NULL, &desc,
        {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
         dev_ctx->get_sampler_desc_heap()->get_element_size() * ((u64)binding_offset)});
    is_bindings_dirty = true;
  }
  void release() override {
    if (signature->num_desc_common) {
      dev_ctx->deferred_release_desc_range(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                           (u32)common_heap_offset, signature->num_desc_common);
    }
    if (signature->num_desc_samplers) {
      dev_ctx->deferred_release_desc_range(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                                           (u32)sampler_heap_offset, signature->num_desc_samplers);
    }
    delete this;
  }
};

rd::IBinding_Table *DX12Device::create_binding_table(Resource_ID signature) {
  SCOPED_LOCK;
  return DX12Binding_Table::create(this, signature_table.load(signature.id));
}

class RenderPass {
  private:
  rd::Render_Pass_Create_Info info;

  public:
  rd::Render_Pass_Create_Info const &get_info() { return info; }
  RenderPass(DX12Device *dev_ctx, rd::Render_Pass_Create_Info const &info) : info(info) {}
  void release() { delete this; }
};
class FrameBuffer {
  private:
  DX12Device *                 dev_ctx    = NULL;
  u32                          rtv_offset = -1;
  i32                          dsv_offset = -1;
  rd::Frame_Buffer_Create_Info info;

  public:
  D3D12_CPU_DESCRIPTOR_HANDLE get_rtv_handle(u32 i) {
    return {dev_ctx->get_rtv_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr +
            (rtv_offset + i) * dev_ctx->get_rtv_desc_heap()->get_element_size()};
  }

  D3D12_CPU_DESCRIPTOR_HANDLE get_dsv_handle() {
    return {dev_ctx->get_dsv_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr +
            rtv_offset * dev_ctx->get_dsv_desc_heap()->get_element_size()};
  }

  rd::Frame_Buffer_Create_Info const &get_info() { return info; }
  FrameBuffer(DX12Device *dev_ctx, rd::Frame_Buffer_Create_Info const &info)
      : dev_ctx(dev_ctx), info(info) {
    if (info.rts.size) {
      rtv_offset = dev_ctx->get_rtv_desc_heap()->allocate(info.rts.size);
      ito(info.rts.size) {
        ID3D12Resource *res = dev_ctx->get_resource(info.rts[i].image.id)->res;
        // Initialize RTVs
        dev_ctx->get_device()->CreateRenderTargetView(res, NULL, get_rtv_handle(i));
      }
    }
    if (info.depth_target.enabled) {
      dsv_offset          = dev_ctx->get_dsv_desc_heap()->allocate(1);
      ID3D12Resource *res = dev_ctx->get_resource(info.depth_target.image.id)->res;
      // Initialize DSV
      dev_ctx->get_device()->CreateDepthStencilView(res, NULL, get_dsv_handle());
    }
  }
  void bind(ComPtr<ID3D12GraphicsCommandList> cmd, RenderPass *pass) {
    InlineArray<D3D12_CPU_DESCRIPTOR_HANDLE, 8> rtvs{};
    ito(info.rts.size) rtvs.push(get_rtv_handle(i));
    if (info.depth_target.enabled) {
      D3D12_CPU_DESCRIPTOR_HANDLE dsv = get_dsv_handle();
      cmd->OMSetRenderTargets(rtvs.size, rtvs.size ? rtvs.elems : NULL, FALSE, &dsv);
    } else {
      cmd->OMSetRenderTargets(rtvs.size, rtvs.size ? rtvs.elems : NULL, FALSE, NULL);
    }
    ito(info.rts.size) {
      // ASSERT_ALWAYS(info.rts[i].format == rd::Format::NATIVE);
      ASSERT_ALWAYS(info.rts[i].layer == 0);
      ASSERT_ALWAYS(info.rts[i].level == 0);
      D3D12_RESOURCE_BARRIER bar{};
      bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      bar.Transition.pResource   = dev_ctx->get_resource(info.rts[i].image.id)->res;
      bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
      bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
      cmd->ResourceBarrier(1, &bar);

      if (pass->get_info().rts[i].clear_color.clear) {
        cmd->ClearRenderTargetView(get_rtv_handle(i),
                                   (float *)&pass->get_info().rts[i].clear_color.r, 0, NULL);
      }
    }
    if (info.depth_target.enabled) {
      // ASSERT_ALWAYS(info.depth_target.format == rd::Format::NATIVE);
      ASSERT_ALWAYS(info.depth_target.layer == 0);
      ASSERT_ALWAYS(info.depth_target.level == 0);
      D3D12_RESOURCE_BARRIER bar{};
      bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      bar.Transition.pResource   = dev_ctx->get_resource(info.depth_target.image.id)->res;
      bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
      bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_DEPTH_WRITE;
      cmd->ResourceBarrier(1, &bar);
      if (pass->get_info().depth_target.clear_depth.clear)
        cmd->ClearDepthStencilView(get_dsv_handle(),
                                   D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                   pass->get_info().depth_target.clear_depth.d, 0, 0, NULL);
    }
  }
  void unbind(ComPtr<ID3D12GraphicsCommandList> cmd) {
    ito(info.rts.size) {
      D3D12_RESOURCE_BARRIER bar{};
      bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      bar.Transition.pResource   = dev_ctx->get_resource(info.rts[i].image.id)->res;
      bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      bar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
      bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
      cmd->ResourceBarrier(1, &bar);
    }
    if (info.depth_target.enabled) {
      D3D12_RESOURCE_BARRIER bar{};
      bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      bar.Transition.pResource   = dev_ctx->get_resource(info.depth_target.image.id)->res;
      bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      bar.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
      bar.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
      cmd->ResourceBarrier(1, &bar);
    }
  }

  void release() {
    if (rtv_offset >= 0)
      dev_ctx->deferred_release_desc_range(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, rtv_offset,
                                           info.rts.size);
    if (dsv_offset >= 0)
      dev_ctx->deferred_release_desc_range(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, dsv_offset,
                                           info.rts.size);
    delete this;
  }
};

class DX12Context : public rd::ICtx {
  ComPtr<ID3D12GraphicsCommandList>              cmd;
  rd::Pass_t                                     type;
  DX12Device *                                   dev_ctx     = NULL;
  DX12Binding_Table *                            cur_binding = NULL;
  Hash_Table<Resource_ID, D3D12_RESOURCE_STATES> resource_state_tracker{};
  RenderPass *                                   render_pass  = NULL;
  FrameBuffer *                                  frame_buffer = NULL;
  GraphicsPSOWrapper *                           gpso         = NULL;
  ~DX12Context()                                              = default;
  // Synchronization
  void _image_barrier(Resource_ID image_id, D3D12_RESOURCE_STATES new_state) {
    InlineArray<D3D12_RESOURCE_BARRIER, 0x10> bars{};
    ID3D12Resource *                          res  = dev_ctx->get_resource(image_id.id)->res;
    auto                                      desc = res->GetDesc();
    if (resource_state_tracker.contains(image_id)) {
    } else {
      resource_state_tracker.insert(image_id, dev_ctx->get_resource(image_id.id)->default_state);
    }
    D3D12_RESOURCE_STATES StateBefore    = resource_state_tracker.get(image_id);
    D3D12_RESOURCE_STATES StateAfter     = new_state;
    resource_state_tracker.get(image_id) = StateAfter;

    u32 num_subresources = desc.DepthOrArraySize * desc.MipLevels;
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) num_subresources = desc.MipLevels;
    // ito(num_subresources) {
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    bar.Transition.pResource   = res;
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    bar.Transition.StateBefore = StateBefore;
    bar.Transition.StateAfter  = StateAfter;
    if (bar.Transition.StateAfter == bar.Transition.StateBefore) {
      bar.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      bar.UAV.pResource = dev_ctx->get_resource(image_id.id)->res;
    }
    bars.push(bar);
    //}
    cmd->ResourceBarrier(bars.size, &bars[0]);
  }
  void _buffer_barrier(Resource_ID buf_id, D3D12_RESOURCE_STATES new_state) {
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    bar.Transition.pResource   = dev_ctx->get_resource(buf_id.id)->res;
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    if (resource_state_tracker.contains(buf_id)) {
    } else {
      resource_state_tracker.insert(buf_id, dev_ctx->get_resource(buf_id.id)->default_state);
    }
    bar.Transition.StateBefore         = resource_state_tracker.get(buf_id);
    bar.Transition.StateAfter          = new_state;
    resource_state_tracker.get(buf_id) = bar.Transition.StateAfter;
    if (bar.Transition.StateAfter == bar.Transition.StateBefore) {
      bar.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      bar.UAV.pResource = dev_ctx->get_resource(buf_id.id)->res;
    }
    cmd->ResourceBarrier(1, &bar);
  }

  public:
  void release() {
    resource_state_tracker.iter_pairs([=](Resource_ID res, D3D12_RESOURCE_STATES state) {
      if (res.type == (u32)Resource_Type::BUFFER) {
        _buffer_barrier(res, dev_ctx->get_resource(res.id)->default_state);
      } else if (res.type == (u32)Resource_Type::TEXTURE) {
        _image_barrier(res, dev_ctx->get_resource(res.id)->default_state);
      } else {
        TRAP;
      }
    });
    if (cur_binding) cur_binding->unbind();
    resource_state_tracker.release();
    delete this;
  }
  DX12Context(DX12Device *device, rd::Pass_t type, RenderPass *render_pass = NULL,
              FrameBuffer *frame_buffer = NULL)
      : type(type), dev_ctx(device), render_pass(render_pass), frame_buffer(frame_buffer) {
    cmd = device->alloc_graphics_cmd();
    device->bind_desc_heaps(cmd.Get());
  }
  void bind_table(rd::IBinding_Table *table) override { //
    if (cur_binding) cur_binding->unbind();
    cur_binding = (DX12Binding_Table *)table;
    cur_binding->bind(cmd, type);
  }
  ComPtr<ID3D12GraphicsCommandList> get_cmd() { return cmd; }
  // Graphics
  void start_render_pass() override {
    ASSERT_DEBUG(render_pass && type == rd::Pass_t::RENDER);
    frame_buffer->bind(cmd, render_pass);
  }
  void end_render_pass() override {
    ASSERT_DEBUG(render_pass && type == rd::Pass_t::RENDER);
    frame_buffer->unbind(cmd);
  }
  void bind_graphics_pso(Resource_ID pso) override {
    GraphicsPSOWrapper *wpso = dev_ctx->get_graphics_pso(pso.id);
    cmd->IASetPrimitiveTopology(wpso->topology);
    cmd->SetPipelineState(wpso->pso);
    gpso = wpso;
  }
  void draw_indexed(u32 indices, u32 instances, u32 first_index, u32 first_instance,
                    i32 vertex_offset) override {
    ASSERT_DEBUG(gpso);
    ASSERT_DEBUG(cur_binding);
    cur_binding->flush_push_constants(cmd, type);
    cmd->DrawIndexedInstanced(indices, instances, first_index, vertex_offset, first_instance);
  }
  void bind_index_buffer(Resource_ID buffer, size_t offset, rd::Index_t format) override {
    ASSERT_DEBUG(gpso);
    ID3D12Resource *        res = dev_ctx->get_resource(buffer.id)->res;
    D3D12_INDEX_BUFFER_VIEW view{};
    view.BufferLocation = res->GetGPUVirtualAddress() + offset;
    view.Format         = to_dx(format);
    view.SizeInBytes    = res->GetDesc().Width - offset;
    cmd->IASetIndexBuffer(&view);
  }
  void bind_vertex_buffer(u32 index, Resource_ID buffer, size_t offset) override {
    ASSERT_DEBUG(gpso);
    ID3D12Resource *         res = dev_ctx->get_resource(buffer.id)->res;
    D3D12_VERTEX_BUFFER_VIEW view{};
    view.BufferLocation = res->GetGPUVirtualAddress() + offset;
    view.SizeInBytes    = res->GetDesc().Width - offset;
    view.StrideInBytes  = gpso->state.bindings[index].stride;
    cmd->IASetVertexBuffers(index, 1, &view);
  }
  void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) override {
    ASSERT_DEBUG(gpso);
    ASSERT_DEBUG(cur_binding);
    cur_binding->flush_push_constants(cmd, type);
    cmd->DrawInstanced(vertices, instances, first_vertex, first_instance);
  }
  void multi_draw_indexed_indirect(Resource_ID arg_buf_id, u32 arg_buf_offset,
                                   Resource_ID cnt_buf_id, u32 cnt_buf_offset, u32 max_count,
                                   u32 stride) override {
    TRAP;
  }
  void set_viewport(float x, float y, float width, float height, float mindepth,
                    float maxdepth) override {
    D3D12_VIEWPORT vp{};
    vp.TopLeftX = x;
    vp.TopLeftY = y;
    vp.Width    = width;
    vp.Height   = height;
    vp.MinDepth = mindepth;
    vp.MaxDepth = maxdepth;
    cmd->RSSetViewports(1, &vp);
  }
  void set_scissor(u32 x, u32 y, u32 width, u32 height) override {
    D3D12_RECT rect{};
    rect.left   = x;
    rect.right  = x + width;
    rect.top    = y;
    rect.bottom = y + height;
    cmd->RSSetScissorRects(1, &rect);
  }
  // Compute
  void bind_compute(Resource_ID id) override {
    ID3D12PipelineState *pso = dev_ctx->get_compute_pso(id.id);
    cmd->SetPipelineState(pso);
  }
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) override {
    ASSERT_DEBUG(cur_binding);
    cur_binding->flush_push_constants(cmd, type);
    cmd->Dispatch(dim_x, dim_y, dim_z);
  }
  // Memory movement
  void fill_buffer(Resource_ID id, size_t offset, size_t size, u32 value) override { TRAP; }
  void clear_image(Resource_ID id, rd::Image_Subresource const &range,
                   rd::Clear_Value const &cv) override {
    TRAP;
  }
  // void update_buffer(Resource_ID buf_id, size_t offset, void const *data,
  //                   size_t data_size) override {
  //
  //}
  void copy_buffer_to_image(Resource_ID buf_id, size_t buffer_offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    u64                                textureMemorySize             = 0;
    constexpr u32                      MAX_TEXTURE_SUBRESOURCE_COUNT = 0x100u;
    u32                                numRows[MAX_TEXTURE_SUBRESOURCE_COUNT];
    u64                                rowSizesInBytes[MAX_TEXTURE_SUBRESOURCE_COUNT];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[MAX_TEXTURE_SUBRESOURCE_COUNT];
    ID3D12Resource *                   tex  = dev_ctx->get_resource(img_id.id)->res;
    D3D12_RESOURCE_DESC                desc = tex->GetDesc();
    u64 numSubResources                     = (u64)desc.MipLevels * (u64)desc.DepthOrArraySize;
    dev_ctx->get_device()->GetCopyableFootprints(&desc, 0, (u32)numSubResources, 0, layouts,
                                                 numRows, rowSizesInBytes, &textureMemorySize);
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource              = dev_ctx->get_resource(buf_id.id)->res;
    src.Type                   = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint        = layouts[dst_info.layer * desc.MipLevels + dst_info.level];
    src.PlacedFootprint.Offset = buffer_offset;
    if (dst_info.buffer_row_pitch)
      src.PlacedFootprint.Footprint.RowPitch = dst_info.buffer_row_pitch;
    src.PlacedFootprint.Footprint.Width  = dst_info.size_x;
    src.PlacedFootprint.Footprint.Height = dst_info.size_y;
    src.PlacedFootprint.Footprint.Depth  = dst_info.size_z;

    if (src.PlacedFootprint.Footprint.Width == 0) src.PlacedFootprint.Footprint.Width = desc.Width;
    if (src.PlacedFootprint.Footprint.Height == 0)
      src.PlacedFootprint.Footprint.Height = desc.Height;
    if (src.PlacedFootprint.Footprint.Depth == 0)
      src.PlacedFootprint.Footprint.Depth = desc.DepthOrArraySize;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource        = tex;
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = dst_info.layer * desc.MipLevels + dst_info.level;
    cmd->CopyTextureRegion(&dst, dst_info.offset_x, dst_info.offset_y, dst_info.offset_z, &src,
                           NULL);
  }
  void copy_image_to_buffer(Resource_ID buf_id, size_t buffer_offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    u64                                textureMemorySize             = 0;
    constexpr u32                      MAX_TEXTURE_SUBRESOURCE_COUNT = 0x100u;
    u32                                numRows[MAX_TEXTURE_SUBRESOURCE_COUNT];
    u64                                rowSizesInBytes[MAX_TEXTURE_SUBRESOURCE_COUNT];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[MAX_TEXTURE_SUBRESOURCE_COUNT];
    ID3D12Resource *                   tex  = dev_ctx->get_resource(img_id.id)->res;
    D3D12_RESOURCE_DESC                desc = tex->GetDesc();
    u64 numSubResources                     = (u64)desc.MipLevels * (u64)desc.DepthOrArraySize;
    dev_ctx->get_device()->GetCopyableFootprints(&desc, 0, (u32)numSubResources, 0, layouts,
                                                 numRows, rowSizesInBytes, &textureMemorySize);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource              = dev_ctx->get_resource(buf_id.id)->res;
    dst.Type                   = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint        = layouts[dst_info.layer * desc.MipLevels + dst_info.level];
    dst.PlacedFootprint.Offset = buffer_offset;

    if (dst_info.buffer_row_pitch)
      dst.PlacedFootprint.Footprint.RowPitch = dst_info.buffer_row_pitch;

    dst.PlacedFootprint.Footprint.Width  = dst_info.size_x;
    dst.PlacedFootprint.Footprint.Height = dst_info.size_y;
    dst.PlacedFootprint.Footprint.Depth  = dst_info.size_z;

    if (dst.PlacedFootprint.Footprint.Width == 0) dst.PlacedFootprint.Footprint.Width = desc.Width;
    if (dst.PlacedFootprint.Footprint.Height == 0)
      dst.PlacedFootprint.Footprint.Height = desc.Height;
    if (dst.PlacedFootprint.Footprint.Depth == 0)
      dst.PlacedFootprint.Footprint.Depth = desc.DepthOrArraySize;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource        = tex;
    src.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = dst_info.layer * desc.MipLevels + dst_info.level;

    D3D12_BOX box{};
    box.left   = dst_info.offset_x;
    box.top    = dst_info.offset_y;
    box.front  = dst_info.offset_z;
    box.right  = box.left + dst_info.size_x;
    box.bottom = box.top + dst_info.size_y;
    box.back   = box.front + dst_info.size_z;

    if (dst_info.size_x == 0) box.right = box.left + desc.Width;
    if (dst_info.size_y == 0) box.bottom = box.top + desc.Height;
    if (dst_info.size_z == 0) box.back = box.front + desc.DepthOrArraySize;

    cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    // D3D12_TEXTURE_COPY_LOCATION dst{};
    // dst.pResource = dev_ctx->get_resource(buf_id.id);
    // dst.Type      = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    // D3D12_TEXTURE_COPY_LOCATION src{};
    // src.pResource = dev_ctx->get_resource(img_id.id);
    // src.Type      = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    //// src.PlacedFootprint.Offset           = 0;
    //// src.PlacedFootprint.Footprint.Width  = dst_info.size_x;
    //// src.PlacedFootprint.Footprint.Height = dst_info.size_y;
    //// src.PlacedFootprint.Footprint.Depth  = dst_info.size_z;

    // cmd->CopyTextureRegion(&dst, buffer_offset, 0, 0, &src, &box);
  }
  void copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                   size_t dst_offset, u32 size) override {
    ID3D12Resource *src = dev_ctx->get_resource(src_buf_id.id)->res;
    ID3D12Resource *dst = dev_ctx->get_resource(dst_buf_id.id)->res;
    cmd->CopyBufferRegion(dst, dst_offset, src, src_offset, size);
  }
  // Synchronization
  void image_barrier(Resource_ID image_id, rd::Image_Access access) override {
    _image_barrier(image_id, to_dx(access));
  }
  void buffer_barrier(Resource_ID buf_id, rd::Buffer_Access access) override {
    _buffer_barrier(buf_id, to_dx(access));
  }
  void insert_event(Resource_ID id) override { TRAP; }
  void insert_timestamp(Resource_ID timestamp_id) override { TRAP; }
};

rd::ICtx *DX12Device::start_compute_pass() {
  SCOPED_LOCK;
  return new DX12Context(this, rd::Pass_t::COMPUTE);
}
void DX12Device::end_compute_pass(rd::ICtx *ctx) {
  SCOPED_LOCK;
  DX12Context *                     d3dctx = ((DX12Context *)ctx);
  ComPtr<ID3D12GraphicsCommandList> cmd    = d3dctx->get_cmd();
  ID3D12CommandList *               icmd   = cmd.Get();
  d3dctx->release();
  DX_ASSERT_OK(cmd->Close());
  cmd_queue->ExecuteCommandLists(1, &icmd);
}
Resource_ID DX12Device::create_graphics_pso(Resource_ID signature, Resource_ID render_pass,
                                            rd::Graphics_Pipeline_State const &state) {
  SCOPED_LOCK;
  RenderPass *                       rp  = renderpass_table.load(render_pass.id);
  DX12Binding_Signature *            sig = signature_table.load(signature.id);
  D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
  ito(rp->get_info().rts.size) {
    desc.BlendState.RenderTarget[i].BlendEnable    = state.blend_states[i].enabled;
    desc.BlendState.RenderTarget[i].BlendOp        = to_dx(state.blend_states[i].color_blend_op);
    desc.BlendState.RenderTarget[i].BlendOpAlpha   = to_dx(state.blend_states[i].alpha_blend_op);
    desc.BlendState.RenderTarget[i].SrcBlend       = to_dx(state.blend_states[i].src_color);
    desc.BlendState.RenderTarget[i].SrcBlendAlpha  = to_dx(state.blend_states[i].src_alpha);
    desc.BlendState.RenderTarget[i].DestBlend      = to_dx(state.blend_states[i].dst_color);
    desc.BlendState.RenderTarget[i].DestBlendAlpha = to_dx(state.blend_states[i].dst_alpha);
    desc.BlendState.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    /* if (rp->get_info().rts[i].format == rd::Format::NATIVE) {
       ID3D12Resource *res = get_resource(rp->get_info().rts[i].image.id);
       desc.RTVFormats[i]  = res->GetDesc().Format;
     } else*/
    desc.RTVFormats[i] = to_dx(rp->get_info().rts[i].format);
  }
  if (rp->get_info().depth_target.enabled) {
    /*if (rp->get_info().depth_target.format == rd::Format::NATIVE) {
      ID3D12Resource *res = get_resource(rp->get_info().depth_target.image.id);
      desc.DSVFormat      = res->GetDesc().Format;
    } else*/
    desc.DSVFormat = to_dx(rp->get_info().depth_target.format);
  }
  desc.SampleDesc.Count   = 1;
  desc.SampleDesc.Quality = 0;
  desc.SampleMask         = UINT_MAX;

  if (state.vs.is_valid()) {
    auto shader             = shader_table.load(state.vs.id);
    desc.VS.BytecodeLength  = shader->GetBufferSize();
    desc.VS.pShaderBytecode = shader->GetBufferPointer();
  }

  if (state.ps.is_valid()) {
    auto shader             = shader_table.load(state.ps.id);
    desc.PS.BytecodeLength  = shader->GetBufferSize();
    desc.PS.pShaderBytecode = shader->GetBufferPointer();
  }

  desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

  desc.DepthStencilState.DepthEnable = state.ds_state.enable_depth_test;
  desc.DepthStencilState.DepthWriteMask =
      state.ds_state.enable_depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
  desc.DepthStencilState.DepthFunc = to_dx(state.ds_state.cmp_op);
  InlineArray<D3D12_INPUT_ELEMENT_DESC, 0x10> ie_desc{};
  ito(state.num_attributes) {
    auto const &             binding = state.bindings[state.attributes[i].binding];
    D3D12_INPUT_ELEMENT_DESC desc{};
    desc.AlignedByteOffset = state.attributes[i].offset;
    desc.Format            = to_dx(state.attributes[i].format);
    desc.InputSlot         = state.attributes[i].binding;
    desc.InputSlotClass    = binding.inputRate == rd::Input_Rate::VERTEX
                              ? D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA
                              : D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA;
    desc.InstanceDataStepRate = binding.inputRate == rd::Input_Rate::VERTEX ? 0 : 1;
    switch (state.attributes[i].type) {
    case rd::Attriute_t::POSITION: {
      desc.SemanticIndex = 0;
      desc.SemanticName  = "POSITION";
      break;
    }
    case rd::Attriute_t::NORMAL: {
      desc.SemanticIndex = 0;
      desc.SemanticName  = "NORMAL";
      break;
    }
    case rd::Attriute_t::BINORMAL: {
      desc.SemanticIndex = 0;
      desc.SemanticName  = "BINORMAL";
      break;
    }
    case rd::Attriute_t::TANGENT: {
      desc.SemanticIndex = 0;
      desc.SemanticName  = "TANGENT";
      break;
    }
    case rd::Attriute_t::TEXCOORD0:
    case rd::Attriute_t::TEXCOORD1:
    case rd::Attriute_t::TEXCOORD2:
    case rd::Attriute_t::TEXCOORD3:
    case rd::Attriute_t::TEXCOORD4:
    case rd::Attriute_t::TEXCOORD5:
    case rd::Attriute_t::TEXCOORD6:
    case rd::Attriute_t::TEXCOORD7: {
      desc.SemanticIndex = (u32)state.attributes[i].type - (u32)rd::Attriute_t::TEXCOORD0;
      desc.SemanticName  = "TEXCOORD";
      break;
    }
    }
    ie_desc.push(desc);
  }
  desc.InputLayout.NumElements        = ie_desc.size;
  desc.InputLayout.pInputElementDescs = &ie_desc[0];
  desc.NumRenderTargets               = state.num_rts;
  desc.PrimitiveTopologyType          = to_dx_type(state.topology);

  desc.RasterizerState.AntialiasedLineEnable = false;
  desc.RasterizerState.ConservativeRaster    = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
  desc.RasterizerState.CullMode              = to_dx(state.rs_state.cull_mode);
  desc.RasterizerState.FillMode              = state.rs_state.polygon_mode == rd::Polygon_Mode::FILL
                                      ? D3D12_FILL_MODE_SOLID
                                      : D3D12_FILL_MODE_WIREFRAME;
  desc.RasterizerState.FrontCounterClockwise =
      state.rs_state.front_face == rd::Front_Face::CW ? true : false;

  desc.pRootSignature = sig->root_signature.Get();

  ID3D12PipelineState *pso = NULL;
  DX_ASSERT_OK(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)));
  GraphicsPSOWrapper *wrapper = new GraphicsPSOWrapper;
  wrapper->state              = state;
  wrapper->topology           = to_dx(state.topology);
  wrapper->pso                = pso;
  return {graphics_pso_table.push(wrapper), (u32)Resource_Type::GRAPHICS_PSO};
}
Resource_ID DX12Device::create_frame_buffer(Resource_ID                         render_pass,
                                            rd::Frame_Buffer_Create_Info const &info) {
  SCOPED_LOCK;
  return {fb_table.push(new FrameBuffer(this, info)), (u32)Resource_Type::FRAME_BUFFER};
}
Resource_ID DX12Device::create_render_pass(rd::Render_Pass_Create_Info const &info) {
  SCOPED_LOCK;
  return {renderpass_table.push(new RenderPass(this, info)), (u32)Resource_Type::RENDER_PASS};
}
rd::ICtx *DX12Device::start_render_pass(Resource_ID render_pass, Resource_ID frame_buffer) {
  SCOPED_LOCK;
  RenderPass * rp = renderpass_table.load(render_pass.id);
  FrameBuffer *fb = fb_table.load(frame_buffer.id);
  return new DX12Context(this, rd::Pass_t::RENDER, rp, fb);
}
void DX12Device::end_render_pass(rd::ICtx *ctx) {
  SCOPED_LOCK;
  DX12Context *                     d3dctx = ((DX12Context *)ctx);
  ComPtr<ID3D12GraphicsCommandList> cmd    = d3dctx->get_cmd();
  ID3D12CommandList *               icmd   = cmd.Get();
  d3dctx->release();
  DX_ASSERT_OK(cmd->Close());
  cmd_queue->ExecuteCommandLists(1, &icmd);
}
void DX12Device::deferred_desc_range_release_iteration() {
  AutoArray<Deferred_Descriptor_Range_Release> new_deferred_desc_range_release;
  new_deferred_desc_range_release.reserve(deferred_desc_range_release.size);
  ito(deferred_desc_range_release.size) {
    auto item = deferred_desc_range_release[i];
    item.timeout -= 1;
    if (item.timeout == 0) {
      switch (item.type) {
      case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV: {
        common_desc_heap->free(item.offset, item.size);
        break;
      }
      case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER: {
        sampler_desc_heap->free(item.offset, item.size);
        break;
      }
      case D3D12_DESCRIPTOR_HEAP_TYPE_RTV: {
        rtv_desc_heap->free(item.offset, item.size);
        break;
      }
      case D3D12_DESCRIPTOR_HEAP_TYPE_DSV: {
        dsv_desc_heap->free(item.offset, item.size);
        break;
      }
      default: {
        TRAP;
      }
      }
    } else {
      new_deferred_desc_range_release.push(item);
    }
  }
  deferred_desc_range_release.reset();
  ito(new_deferred_desc_range_release.size) {
    deferred_desc_range_release.push(new_deferred_desc_range_release[i]);
  }
}
void DX12Device::deferred_resource_release_iteration() {
  AutoArray<Pair<Resource_ID, i32>> new_deferred_release;
  ito(deferred_release.size) {
    auto item = deferred_release[i];
    item.second -= 1;
    if (item.second == 0) {
      if (item.first.type == (u32)Resource_Type::BUFFER ||
          item.first.type == (u32)Resource_Type::TEXTURE) {
        auto wr = resource_table.load(item.first.id);
        wr->res->Release();
        delete wr;
        resource_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::RENDER_PASS) {
        auto render_pass = renderpass_table.load(item.first.id);
        render_pass->release();
        renderpass_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::FRAME_BUFFER) {
        auto fb = fb_table.load(item.first.id);
        fb->release();
        fb_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::COMPUTE_PSO) {
        auto pso = compute_pso_table.load(item.first.id);
        pso->Release();
        compute_pso_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::GRAPHICS_PSO) {
        auto pso = graphics_pso_table.load(item.first.id);
        pso->pso->Release();
        delete pso;
        graphics_pso_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::SIGNATURE) {
        auto sig = signature_table.load(item.first.id);
        sig->release();
        signature_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::SHADER) {
        auto sig = shader_table.load(item.first.id);
        sig->Release();
        shader_table.free(item.first.id);
      } else if (item.first.type == (u32)Resource_Type::SAMPLER) {
        auto s = sampler_table.load(item.first.id);
        delete s;
        sampler_table.free(item.first.id);
      } else {
        TRAP;
      }
    } else {
      new_deferred_release.push(item);
    }
  }
  deferred_release.reset();
  ito(new_deferred_release.size) { deferred_release.push(new_deferred_release[i]); }
}
} // namespace

namespace rd {
rd::IDevice *create_dx12(void *window_handler) { return new DX12Device(window_handler); }
} // namespace rd