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
DXGI_FORMAT to_dx(rd::Format format) {
  // clang-format off
  switch (format) {
  case rd::Format::RGBA8_UNORM     : return DXGI_FORMAT_R8G8B8A8_UNORM      ;
  case rd::Format::RGBA8_SNORM     : return DXGI_FORMAT_R8G8B8A8_SNORM      ;
  case rd::Format::RGBA8_SRGBA     : return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB       ;
  case rd::Format::RGBA8_UINT      : return DXGI_FORMAT_R8G8B8A8_UINT;

  case rd::Format::BGRA8_SRGBA     : return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB       ;

  //case rd::Format::RGB8_UNORM      : return DXGI_FORMAT_R8G8B8_UNORM        ;
  //case rd::Format::RGB8_SNORM      : return DXGI_FORMAT_R8G8B8_SNORM        ;
  //case rd::Format::RGB8_SRGBA      : return VK_FORMAT_R8G8B8_SRGB         ;
  //case rd::Format::RGB8_UINT       : return VK_FORMAT_R8G8B8_UINT         ;

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

D3D12_RESOURCE_STATES get_default_state(ID3D12Resource *res) {
  D3D12_HEAP_PROPERTIES hp{};
  D3D12_HEAP_FLAGS      hf{};
  DX_ASSERT_OK(res->GetHeapProperties(&hp, &hf));
  if (hp.Type == D3D12_HEAP_TYPE_READBACK) return D3D12_RESOURCE_STATE_COPY_DEST;
  if (hp.Type == D3D12_HEAP_TYPE_UPLOAD) return D3D12_RESOURCE_STATE_GENERIC_READ;
  if (res->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    return D3D12_RESOURCE_STATE_COMMON;
  return D3D12_RESOURCE_STATE_COMMON;
}

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
  GPU_Desc_Heap(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type, u32 size)
      : ual(size), type(type) {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type                       = type;
    desc.NumDescriptors             = size;
    desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));
    element_size = device->GetDescriptorHandleIncrementSize(type);
  }
  ~GPU_Desc_Heap() {}
  i32 allocate(u32 size) {
    std::lock_guard<std::mutex> _lock(mutex);
    return ual.alloc(size);
  }
  void free(u32 offset, u32 size) {
    std::lock_guard<std::mutex> _lock(mutex);
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
  PSO,
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

  i32 common_param_id   = -1;
  i32 samplers_param_id = -1;
  i32 pc_param_id       = -1;
  u32 num_params        = 0;
  // u32                         num_params_common   = 0;
  // u32                         num_params_samplers = 0;
  // u32                         num_params_total    = 0;
  u32 push_constants_size = 0;
  // InlineArray<u32, 0x10>      common_parameter_heap_offsets{};
  // InlineArray<u32, 0x10>      common_parameter_space_to_parameter_id{};
  // InlineArray<u32, 0x10>      sampler_parameter_heap_offsets{};
  // InlineArray<u32, 0x10>      sampler_parameter_space_to_parameter_id{};
  enum { PUSH_CONSTANTS_SPACE = 777 };
  struct ParamDesc {
    InlineArray<D3D12_DESCRIPTOR_RANGE, 0x10> bindings{};
  };
  ParamDesc space_descs[0x10]{};

  static DX12Binding_Signature *create(DX12Device *                         dev_ctx,
                                       rd::Binding_Table_Create_Info const &table_info);
};

class DX12Device : public rd::IDevice {
  static constexpr u32 NUM_BACK_BUFFERS = 3;
  static constexpr u32 MAX_THREADS      = 0x100;

  ComPtr<ID3D12Device2>          device;
  ComPtr<ID3D12CommandQueue>     cmd_queue;
  ComPtr<IDXGISwapChain4>        sc;
  ComPtr<ID3D12Resource>         sc_images[NUM_BACK_BUFFERS];
  ComPtr<ID3D12CommandAllocator> cmd_allocs[NUM_BACK_BUFFERS];
  ComPtr<ID3D12DescriptorHeap>   rtv_desc_heap;
  ComPtr<ID3D12Fence>            fence;
  HANDLE                         fence_event = 0;
  ComPtr<ID3D12Resource>         main_rt[NUM_BACK_BUFFERS];
  D3D12_CPU_DESCRIPTOR_HANDLE    main_rt_desc[NUM_BACK_BUFFERS];
  HANDLE                         sc_wait_obj       = 0;
  u32                            cur_cmd_id        = 0;
  GPU_Desc_Heap *                common_desc_heap  = NULL;
  GPU_Desc_Heap *                sampler_desc_heap = NULL;
  std::mutex                     mutex;

  Resource_Array<ID3D12Resource *>        resource_table;
  Resource_Array<DX12Binding_Signature *> signature_table;
  Resource_Array<IDxcBlob *>              shader_table;
  Resource_Array<ID3D12PipelineState *>   pso_table;
  Resource_Array<HANDLE>                  events_table;
  Resource_Array<D3D12_SAMPLER_DESC *>    sampler_table;

  AutoArray<Pair<Resource_ID, i32>> deferred_release;

  ~DX12Device() {}

  // Synchronization
  void sync_transition_barrier(ID3D12Resource *res, D3D12_RESOURCE_STATES StateBefore,
                               D3D12_RESOURCE_STATES StateAfter) {
    auto                   cmd = alloc_graphics_cmd();
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    bar.Transition.pResource   = res;
    bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = StateBefore;
    bar.Transition.StateAfter  = StateAfter;
    cmd->ResourceBarrier(1, &bar);
    cmd->Close();
    cmd_queue->ExecuteCommandLists(1, (ID3D12CommandList **)cmd.GetAddressOf());
    wait_idle();
  }

  public:
  ID3D12Resource *       get_resource(ID id) { return resource_table.load(id); }
  DX12Binding_Signature *get_signature(ID id) { return signature_table.load(id); }
  ID3D12PipelineState *  get_pso(ID id) { return pso_table.load(id); }
  D3D12_SAMPLER_DESC *   get_sampler(ID id) { return sampler_table.load(id); }

  ComPtr<ID3D12Device2>             get_device() { return device; }
  ComPtr<ID3D12GraphicsCommandList> alloc_graphics_cmd() {
    ComPtr<ID3D12GraphicsCommandList> out;
    DX_ASSERT_OK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           cmd_allocs[cur_cmd_id].Get(), NULL, IID_PPV_ARGS(&out)));
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
  DX12Device(void *hdl) {
    {
      ComPtr<ID3D12Debug> debugInterface;
      DX_ASSERT_OK(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
      debugInterface->EnableDebugLayer();
    }
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
    DX_ASSERT_OK(D3D12CreateDevice(NULL, featureLevel, IID_PPV_ARGS(&device)));
    sampler_desc_heap = new GPU_Desc_Heap(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048);
    common_desc_heap =
        new GPU_Desc_Heap(device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1 << 19);

    {
      D3D12_DESCRIPTOR_HEAP_DESC desc = {};
      desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
      desc.NumDescriptors             = NUM_BACK_BUFFERS;
      desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      desc.NodeMask                   = 1;
      DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&rtv_desc_heap)));

      SIZE_T rtvDescriptorSize =
          device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
      D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtv_desc_heap->GetCPUDescriptorHandleForHeapStart();
      for (UINT i = 0; i < NUM_BACK_BUFFERS; i++) {
        main_rt_desc[i] = rtvHandle;
        rtvHandle.ptr += rtvDescriptorSize;
      }
    }
    {
      D3D12_COMMAND_QUEUE_DESC desc = {};
      desc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
      desc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
      desc.NodeMask                 = 1;
      DX_ASSERT_OK(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&cmd_queue)));
    }

    ito(NUM_BACK_BUFFERS) DX_ASSERT_OK(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmd_allocs[i])));

    DX_ASSERT_OK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

    fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    ASSERT_ALWAYS(fence_event);
    if (hdl) {
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
      DX_ASSERT_OK(swapChain1->QueryInterface(IID_PPV_ARGS(&sc)));

      sc->SetMaximumFrameLatency(NUM_BACK_BUFFERS);
      sc_wait_obj = sc->GetFrameLatencyWaitableObject();
    }
  }
  Resource_ID create_image(rd::Image_Create_Info info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    ID3D12Resource *            buf = NULL;
    D3D12_HEAP_PROPERTIES       prop{};
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
    return {resource_table.push(buf), (u32)Resource_Type::TEXTURE};
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
      std::lock_guard<std::mutex> _lock(mutex);
      DX_ASSERT_OK(
          device->CreateCommittedResource(&prop, flags, &desc, state, NULL, IID_PPV_ARGS(&buf)));
      ASSERT_DEBUG(get_default_state(buf) == state);
    }
    /*if (state != D3D12_RESOURCE_STATE_COMMON) {
      sync_transition_barrier(buf, state, D3D12_RESOURCE_STATE_COMMON);
    }*/
    return {resource_table.push(buf), (u32)Resource_Type::BUFFER};
  }
  Resource_ID create_shader(rd::Stage_t type, string_ref text,
                            Pair<string_ref, string_ref> *defines, size_t num_defines) override {
    std::lock_guard<std::mutex> _lock(mutex);
    static ComPtr<IDxcLibrary>  library;
    static ComPtr<IDxcCompiler> compiler;
    static int                  _init = [] {
      DX_ASSERT_OK(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library)));
      DX_ASSERT_OK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));
      return 0;
    }();
    ComPtr<IDxcBlobEncoding> blob;
    // Do a little preprocessing
    {
      TMP_STORAGE_SCOPE;
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
    }
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
    HRESULT                     hr = compiler->Compile(blob.Get(),     // pSource
                                   L"shader.hlsl", // pSourceName
                                   L"main",        // pEntryPoint
                                   profile,        // pTargetProfile
                                   options, (u32)ARRAYSIZE(options) - 1, // pArguments, argCount
                                   dxc_defines.ptr, (u32)dxc_defines.size, // pDefines, defineCount
                                   NULL,     // pIncludeHandler
                                   &result); // ppResult
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
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) override {
    std::lock_guard<std::mutex> _lock(mutex);
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
    std::lock_guard<std::mutex> _lock(mutex);
    deferred_release.push({id, 3});
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
       std::lock_guard<std::mutex> _lock(mutex);
       DX_ASSERT_OK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
     }
     HANDLE event = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
     defer(CloseHandle(event));
     DX_ASSERT_OK(fence->SetEventOnCompletion(1, event));
     {
       std::lock_guard<std::mutex> _lock(mutex);
       cmd_queue->Signal(fence.Get(), 1);
     }
     WaitForSingleObject(event, INFINITE);*/
    TRAP;
  }
  Resource_ID create_timestamp() override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID get_swapchain_image() override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::Image2D_Info get_swapchain_image_info() override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::Image_Info get_image_info(Resource_ID res_id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  void *map_buffer(Resource_ID id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    ID3D12Resource *            res = get_resource(id.id);
    D3D12_RANGE                 rr{};
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
    std::lock_guard<std::mutex> _lock(mutex);
    ID3D12Resource *            res = get_resource(id.id);
    D3D12_RANGE                 rr{};
    rr.Begin                 = 0;
    rr.End                   = 0;
    D3D12_RESOURCE_DESC desc = res->GetDesc();
    ASSERT_DEBUG(desc.Format = DXGI_FORMAT_R32_TYPELESS);
    rr.End = desc.Width;
    res->Unmap(0, &rr);
  }
  Resource_ID create_render_pass(rd::Render_Pass_Create_Info const &info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_compute_pso(Resource_ID signature, Resource_ID cs) override {
    std::lock_guard<std::mutex>       _lock(mutex);
    IDxcBlob *                        bytecode = shader_table.load(cs.id);
    DX12Binding_Signature *           sig      = signature_table.load(signature.id);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.CS.pShaderBytecode  = bytecode->GetBufferPointer();
    desc.CS.BytecodeLength   = bytecode->GetBufferSize();
    desc.pRootSignature      = sig->root_signature.Get();
    desc.Flags               = D3D12_PIPELINE_STATE_FLAG_NONE;
    ID3D12PipelineState *pso = NULL;
    DX_ASSERT_OK(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)));
    return {pso_table.push(pso), (u32)Resource_Type::PSO};
  }
  Resource_ID create_graphics_pso(Resource_ID signature, Resource_ID render_pass,
                                  rd::Graphics_Pipeline_State const &) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_signature(rd::Binding_Table_Create_Info const &info) override {
    return {signature_table.push(DX12Binding_Signature::create(this, info)),
            (u32)Resource_Type::SIGNATURE};
  }
  rd::IBinding_Table *create_binding_table(Resource_ID signature) override;
  rd::ICtx *          start_render_pass(Resource_ID render_pass) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  void end_render_pass(rd::ICtx *ctx) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::ICtx *start_compute_pass() override;
  void      end_compute_pass(rd::ICtx *ctx) override;
  bool      get_timestamp_state(Resource_ID) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  double get_timestamp_ms(Resource_ID t0, Resource_ID t1) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  void wait_idle() override {
    ComPtr<ID3D12Fence> fence;
    {
      std::lock_guard<std::mutex> _lock(mutex);
      DX_ASSERT_OK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }
    HANDLE event = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
    defer(CloseHandle(event));
    DX_ASSERT_OK(fence->SetEventOnCompletion(1, event));
    {
      std::lock_guard<std::mutex> _lock(mutex);
      cmd_queue->Signal(fence.Get(), 1);
    }
    WaitForSingleObject(event, INFINITE);
  }
  bool get_event_state(Resource_ID id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::Impl_t getImplType() override { TRAP; }
  void       release() override {
    wait_idle();
    if (common_desc_heap) delete common_desc_heap;
    if (sampler_desc_heap) delete sampler_desc_heap;
    delete this;
  }
  void start_frame() override {
    std::lock_guard<std::mutex>       _lock(mutex);
    AutoArray<Pair<Resource_ID, i32>> new_deferred_release;
    ito(deferred_release.size) {
      auto item = deferred_release[i];
      item.second -= 1;
      if (item.second == 0) {
        if (item.first.type == (u32)Resource_Type::BUFFER ||
            item.first.type == (u32)Resource_Type::TEXTURE) {
          ID3D12Resource *res = resource_table.load(item.first.id);
          res->Release();
          resource_table.free(item.first.id);
        } else {
          TRAP;
        }
      } else {
        new_deferred_release.push(item);
      }
    }
    deferred_release.reset();
    ito(new_deferred_release.size) { deferred_release.push(new_deferred_release[i]); }
    cmd_allocs[cur_cmd_id]->Reset();
  }
  void end_frame() override { std::lock_guard<std::mutex> _lock(mutex); }
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
    desc.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ID3DBlob *blob                 = NULL;
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
  void flush_bindings(ComPtr<ID3D12GraphicsCommandList> cmd) {
    cmd->SetComputeRootSignature(signature->root_signature.Get());
    if (signature->common_param_id >= 0) {
      dev_ctx->get_device()->CopyDescriptorsSimple(
          signature->num_desc_common,
          {dev_ctx->get_common_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_common_desc_heap()->get_element_size() * common_heap_offset},
          cpu_common_heap->GetCPUDescriptorHandleForHeapStart(),
          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      cmd->SetComputeRootDescriptorTable(
          signature->common_param_id,
          {dev_ctx->get_common_desc_heap()->get_heap()->GetGPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_common_desc_heap()->get_element_size() * common_heap_offset});
    }
    if (signature->samplers_param_id >= 0) {
      auto heap_start =
          dev_ctx->get_sampler_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr;
      auto stride = dev_ctx->get_sampler_desc_heap()->get_element_size();
      dev_ctx->get_device()->CopyDescriptorsSimple(
          signature->num_desc_samplers, {heap_start + stride * sampler_heap_offset},
          cpu_sampler_heap->GetCPUDescriptorHandleForHeapStart(),
          D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
      cmd->SetComputeRootDescriptorTable(
          signature->samplers_param_id,
          {dev_ctx->get_sampler_desc_heap()->get_heap()->GetGPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_sampler_desc_heap()->get_element_size() * sampler_heap_offset});
    }
  }
  void flush_push_constants(ComPtr<ID3D12GraphicsCommandList> cmd) {
    if (signature->push_constants_size)
      cmd->SetComputeRoot32BitConstants(signature->pc_param_id, signature->push_constants_size / 4,
                                        push_constants_data, 0);
  }
  void push_constants(void const *data, size_t offset, size_t size) override {
    memcpy(push_constants_data + offset, data, size);
  }
  // u32  get_push_constants_parameter_id() { return num_params_total - 1; }
  // u32  get_push_constants_size() { return push_constants_size; }
  void bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                    size_t size) override {
    ID3D12Resource *res = dev_ctx->get_resource(buf_id.id);
    u32             binding_offset =
        signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart;
    D3D12_CONSTANT_BUFFER_VIEW_DESC desc{};
    ASSERT_DEBUG((offset & 0xffu) == 0);
    desc.BufferLocation = res->GetGPUVirtualAddress() + offset;
    desc.SizeInBytes    = (size + 0xffU) & ~0xffu;
    dev_ctx->get_device()->CreateConstantBufferView(
        &desc, {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
                dev_ctx->get_common_desc_heap()->get_element_size() * ((u64)binding_offset)});
  }
  void bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) override {
    D3D12_SAMPLER_DESC *desc = dev_ctx->get_sampler(sampler_id.id);
    u32 offset = signature->space_descs[space].bindings[binding].OffsetInDescriptorsFromTableStart;
    dev_ctx->get_device()->CreateSampler(
        desc, {cpu_sampler_heap->GetCPUDescriptorHandleForHeapStart().ptr +
               dev_ctx->get_sampler_desc_heap()->get_element_size() * ((u64)offset)});
  }
  void bind_UAV_buffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                       size_t size) override {
    ID3D12Resource *res = dev_ctx->get_resource(buf_id.id);
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
  }
  void bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                    rd::Image_Subresource const &_range, rd::Format format) override {
    ID3D12Resource *res = dev_ctx->get_resource(image_id.id);
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
  }
  void bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                        rd::Image_Subresource const &_range, rd::Format format) override {
    ID3D12Resource *res = dev_ctx->get_resource(image_id.id);
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
  }
  void release() override {
    if (signature->num_desc_common) {
      dev_ctx->get_common_desc_heap()->free((u32)common_heap_offset, signature->num_desc_common);
    }
    if (signature->num_desc_samplers) {
      dev_ctx->get_sampler_desc_heap()->free((u32)sampler_heap_offset,
                                             signature->num_desc_samplers);
    }
    delete this;
  }
  void clear_bindings() override { TRAP; }
};

rd::IBinding_Table *DX12Device::create_binding_table(Resource_ID signature) {
  std::lock_guard<std::mutex> _lock(mutex);
  return DX12Binding_Table::create(this, signature_table.load(signature.id));
}

class DX12Context : public rd::ICtx {
  ComPtr<ID3D12GraphicsCommandList>              cmd;
  rd::Pass_t                                     type;
  DX12Device *                                   dev_ctx     = NULL;
  DX12Binding_Table *                            cur_binding = NULL;
  Hash_Table<Resource_ID, D3D12_RESOURCE_STATES> resource_state_tracker{};
  ~DX12Context() = default;
  // Synchronization
  void _image_barrier(Resource_ID image_id, D3D12_RESOURCE_STATES new_state) {
    InlineArray<D3D12_RESOURCE_BARRIER, 0x10> bars{};
    ID3D12Resource *                          res  = dev_ctx->get_resource(image_id.id);
    auto                                      desc = res->GetDesc();
    if (resource_state_tracker.contains(image_id)) {
    } else {
      resource_state_tracker.insert(image_id, get_default_state(res));
    }
    D3D12_RESOURCE_STATES StateBefore    = resource_state_tracker.get(image_id);
    D3D12_RESOURCE_STATES StateAfter     = new_state;
    resource_state_tracker.get(image_id) = StateAfter;

    u32 num_subresources = desc.DepthOrArraySize * desc.MipLevels;
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) num_subresources = desc.MipLevels;
    ito(num_subresources) {
      D3D12_RESOURCE_BARRIER bar{};
      bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      bar.Transition.pResource   = res;
      bar.Transition.Subresource = i;
      bar.Transition.StateBefore = StateBefore;
      bar.Transition.StateAfter  = StateAfter;
      if (bar.Transition.StateAfter == bar.Transition.StateBefore) {
        bar.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        bar.UAV.pResource = dev_ctx->get_resource(image_id.id);
      }
      bars.push(bar);
    }
    cmd->ResourceBarrier(bars.size, &bars[0]);
  }
  void _buffer_barrier(Resource_ID buf_id, D3D12_RESOURCE_STATES new_state) {
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    bar.Transition.pResource   = dev_ctx->get_resource(buf_id.id);
    bar.Transition.Subresource = 0;
    if (resource_state_tracker.contains(buf_id)) {
    } else {
      resource_state_tracker.insert(buf_id, get_default_state(bar.Transition.pResource));
    }
    bar.Transition.StateBefore         = resource_state_tracker.get(buf_id);
    bar.Transition.StateAfter          = new_state;
    resource_state_tracker.get(buf_id) = bar.Transition.StateAfter;
    if (bar.Transition.StateAfter == bar.Transition.StateBefore) {
      bar.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      bar.UAV.pResource = dev_ctx->get_resource(buf_id.id);
    }
    cmd->ResourceBarrier(1, &bar);
  }

  public:
  void release() {
    resource_state_tracker.iter_pairs([=](Resource_ID res, D3D12_RESOURCE_STATES state) {
      if (res.type == (u32)Resource_Type::BUFFER) {
        _buffer_barrier(res, get_default_state(dev_ctx->get_resource(res.id)));
      } else if (res.type == (u32)Resource_Type::TEXTURE) {
        _image_barrier(res, get_default_state(dev_ctx->get_resource(res.id)));
      } else {
        TRAP;
      }
    });
    resource_state_tracker.release();
    delete this;
  }
  DX12Context(DX12Device *device, rd::Pass_t type) : type(type), dev_ctx(device) {
    cmd = device->alloc_graphics_cmd();
    device->bind_desc_heaps(cmd.Get());
  }
  void bind_table(rd::IBinding_Table *table) override { //
    cur_binding = (DX12Binding_Table *)table;
  }
  ComPtr<ID3D12GraphicsCommandList> get_cmd() { return cmd; }
  // Graphics
  void start_render_pass() override { TRAP; }
  void end_render_pass() override { TRAP; }
  void bind_graphics_pso(Resource_ID pso) override { TRAP; }
  void draw_indexed(u32 indices, u32 instances, u32 first_index, u32 first_instance,
                    i32 vertex_offset) override {
    TRAP;
  }
  void bind_index_buffer(Resource_ID id, u32 offset, rd::Index_t format) override { TRAP; }
  void bind_vertex_buffer(u32 index, Resource_ID buffer, size_t offset) override { TRAP; }
  void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) override { TRAP; }
  void multi_draw_indexed_indirect(Resource_ID arg_buf_id, u32 arg_buf_offset,
                                   Resource_ID cnt_buf_id, u32 cnt_buf_offset, u32 max_count,
                                   u32 stride) override {
    TRAP;
  }

  void set_viewport(float x, float y, float width, float height, float mindepth,
                    float maxdepth) override {
    TRAP;
  }
  void set_scissor(u32 x, u32 y, u32 width, u32 height) override { TRAP; }
  void RS_set_line_width(float width) override { TRAP; }
  void RS_set_depth_bias(float width) override { TRAP; }
  // Compute
  void bind_compute(Resource_ID id) override {
    ID3D12PipelineState *pso = dev_ctx->get_pso(id.id);
    cmd->SetPipelineState(pso);
  }
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) override {
    ASSERT_DEBUG(cur_binding);
    cur_binding->flush_bindings(cmd);
    cur_binding->flush_push_constants(cmd);
    cmd->Dispatch(dim_x, dim_y, dim_z);
  }
  // Memory movement
  void fill_buffer(Resource_ID id, size_t offset, size_t size, u32 value) override { TRAP; }
  void clear_image(Resource_ID id, rd::Image_Subresource const &range,
                   rd::Clear_Value const &cv) override {
    TRAP;
  }
  void update_buffer(Resource_ID buf_id, size_t offset, void const *data,
                     size_t data_size) override {
    TRAP;
  }
  void copy_buffer_to_image(Resource_ID buf_id, size_t buffer_offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    u64                                textureMemorySize             = 0;
    constexpr u32                      MAX_TEXTURE_SUBRESOURCE_COUNT = 0x100u;
    u32                                numRows[MAX_TEXTURE_SUBRESOURCE_COUNT];
    u64                                rowSizesInBytes[MAX_TEXTURE_SUBRESOURCE_COUNT];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[MAX_TEXTURE_SUBRESOURCE_COUNT];
    ID3D12Resource *                   tex  = dev_ctx->get_resource(img_id.id);
    D3D12_RESOURCE_DESC                desc = tex->GetDesc();
    u64 numSubResources                     = (u64)desc.MipLevels * (u64)desc.DepthOrArraySize;
    dev_ctx->get_device()->GetCopyableFootprints(&desc, 0, (u32)numSubResources, 0, layouts,
                                                 numRows, rowSizesInBytes, &textureMemorySize);
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource              = dev_ctx->get_resource(buf_id.id);
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
    ID3D12Resource *                   tex  = dev_ctx->get_resource(img_id.id);
    D3D12_RESOURCE_DESC                desc = tex->GetDesc();
    u64 numSubResources                     = (u64)desc.MipLevels * (u64)desc.DepthOrArraySize;
    dev_ctx->get_device()->GetCopyableFootprints(&desc, 0, (u32)numSubResources, 0, layouts,
                                                 numRows, rowSizesInBytes, &textureMemorySize);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource              = dev_ctx->get_resource(buf_id.id);
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
    ID3D12Resource *src = dev_ctx->get_resource(src_buf_id.id);
    ID3D12Resource *dst = dev_ctx->get_resource(dst_buf_id.id);
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
  std::lock_guard<std::mutex> _lock(mutex);
  return new DX12Context(this, rd::Pass_t::COMPUTE);
}
void DX12Device::end_compute_pass(rd::ICtx *ctx) {
  std::lock_guard<std::mutex>       _lock(mutex);
  DX12Context *                     d3dctx = ((DX12Context *)ctx);
  ComPtr<ID3D12GraphicsCommandList> cmd    = d3dctx->get_cmd();
  ID3D12CommandList *               icmd   = cmd.Get();
  d3dctx->release();
  cmd->Close();
  cmd_queue->ExecuteCommandLists(1, &icmd);
}
} // namespace

namespace rd {
rd::IDevice *create_dx12(void *window_handler) { return new DX12Device(window_handler); }
} // namespace rd