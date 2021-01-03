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
  void free(ID id) { free_slots.push(id.index()); }
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
  u32                         num_desc_common     = 0;
  u32                         num_desc_samplers   = 0;
  u32                         num_params_common   = 0;
  u32                         num_params_samplers = 0;
  u32                         num_params_total    = 0;
  u32                         push_constants_size = 0;
  InlineArray<u32, 0x10>      common_parameter_heap_offsets{};
  InlineArray<u32, 0x10>      sampler_parameter_heap_offsets{};
  enum { PUSH_CONSTANTS_SPACE = 777 };
  struct SpaceDesc {
    InlineArray<D3D12_DESCRIPTOR_RANGE, 0x10> ranges{};
  };
  // 16 for common and 16 for samplers
  SpaceDesc space_descs[0x20]{};

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

  AutoArray<Pair<Resource_ID, i32>> deferred_release;

  ~DX12Device() {}

  public:
  ID3D12Resource *       get_resource(ID id) { return resource_table.load(id); }
  DX12Binding_Signature *get_signature(ID id) { return signature_table.load(id); }
  ID3D12PipelineState *  get_pso(ID id) { return pso_table.load(id); }

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
    TRAP;
  }
  Resource_ID create_buffer(rd::Buffer_Create_Info info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    ID3D12Resource *            buf   = NULL;
    D3D12_HEAP_PROPERTIES       prop  = get_heap_properties(info.memory_type);
    D3D12_HEAP_FLAGS            flags = D3D12_HEAP_FLAG_NONE;
    D3D12_RESOURCE_DESC         desc{};
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
    if (info.memory_type == rd::Memory_Type::CPU_READ_WRITE) state = D3D12_RESOURCE_STATE_COPY_DEST;
    DX_ASSERT_OK(
        device->CreateCommittedResource(&prop, flags, &desc, state, NULL, IID_PPV_ARGS(&buf)));
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
    DX_ASSERT_OK(library->CreateBlobWithEncodingFromPinned(text.ptr, (uint32_t)text.len, 0, &blob));
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
    WCHAR const *               options[] = {L"UNUSED"};
    ComPtr<IDxcOperationResult> result;
    HRESULT                     hr = compiler->Compile(blob.Get(),     // pSource
                                   L"shader.hlsl", // pSourceName
                                   L"main",        // pEntryPoint
                                   profile,        // pTargetProfile
                                   options, ARRAYSIZE(options) - 1, // pArguments, argCount
                                   dxc_defines.ptr, dxc_defines.size, // pDefines, defineCount
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
    TRAP;
  }
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
  DX12Binding_Signature *out                     = new DX12Binding_Signature;
  out->push_constants_size                       = table_info.push_constants_size;
  auto                                    device = dev_ctx->get_device();
  InlineArray<D3D12_ROOT_PARAMETER, 0x20> params{};
  out->num_params_common = 0;
  ito(table_info.spaces.size) {
    auto const &info = table_info.spaces[i];
    out->common_parameter_heap_offsets.push(out->num_desc_common);
    jto(info.bindings.size) {
      auto const &b = info.bindings[j];
      // Common descriptors only
      if (b.type == rd::Binding_t::SAMPLER) continue;

      D3D12_DESCRIPTOR_RANGE desc_range            = {};
      desc_range.RangeType                         = to_dx(b.type);
      desc_range.NumDescriptors                    = b.num_array_elems;
      desc_range.BaseShaderRegister                = b.binding;
      desc_range.RegisterSpace                     = i;
      desc_range.OffsetInDescriptorsFromTableStart = out->num_desc_common;
      out->num_desc_common = MAX(out->num_desc_common, b.binding + b.num_array_elems);
      out->space_descs[i].ranges.push(desc_range);
    }
    D3D12_ROOT_PARAMETER param                = {};
    param.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = out->space_descs[i].ranges.size;
    param.DescriptorTable.pDescriptorRanges   = &out->space_descs[i].ranges[0];
    param.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
    params.push(param);
    out->num_params_common++;
  }
  out->num_params_samplers = 0;
  // ito(table_info.spaces.size) {
  //  auto const &info = table_info.spaces[i];
  //  out->sampler_parameter_heap_offsets.push(out->num_desc_samplers);
  //  jto(info.bindings.size) {
  //    auto const &b = info.bindings[j];
  //    // Sampler descriptors only
  //    if (b.type != rd::Binding_t::SAMPLER) continue;

  //    D3D12_DESCRIPTOR_RANGE desc_range            = {};
  //    desc_range.RangeType                         = to_dx(b.type);
  //    desc_range.NumDescriptors                    = b.num_array_elems;
  //    desc_range.BaseShaderRegister                = b.binding;
  //    desc_range.RegisterSpace                     = i;
  //    desc_range.OffsetInDescriptorsFromTableStart = out->num_desc_samplers;
  //    out->num_desc_samplers = MAX(out->num_desc_samplers, b.binding + b.num_array_elems);
  //    out->space_descs[i + 0x10].ranges.push(desc_range);
  //  }
  //  D3D12_ROOT_PARAMETER param                = {};
  //  param.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  //  param.DescriptorTable.NumDescriptorRanges = out->space_descs[i + 0x10].ranges.size;
  //  param.DescriptorTable.pDescriptorRanges   = &out->space_descs[i + 0x10].ranges[0];
  //  param.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
  //  params.push(param);
  //  out->num_params_samplers++;
  //}
  if (table_info.push_constants_size > 0) {
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants.ShaderRegister = 0;
    param.Constants.RegisterSpace  = PUSH_CONSTANTS_SPACE;
    param.Constants.Num32BitValues = table_info.push_constants_size / 4;
    param.ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;
    params.push(param);
  }
  out->num_params_total          = params.size;
  D3D12_ROOT_SIGNATURE_DESC desc = {};
  desc.NumParameters             = params.size;
  desc.pParameters               = &params[0];
  desc.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_NONE;
  ID3DBlob *blob                 = NULL;
  DX_ASSERT_OK(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL));
  DX_ASSERT_OK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                           IID_PPV_ARGS(&out->root_signature)));
  blob->Release();
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
    if (signature->num_desc_common) {
      dev_ctx->get_device()->CopyDescriptorsSimple(
          signature->num_desc_common,
          {dev_ctx->get_common_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_common_desc_heap()->get_element_size() * common_heap_offset},
          cpu_common_heap->GetCPUDescriptorHandleForHeapStart(),
          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      ito(signature->num_params_common) cmd->SetComputeRootDescriptorTable(
          i,
          {dev_ctx->get_common_desc_heap()->get_heap()->GetGPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_common_desc_heap()->get_element_size() *
               (common_heap_offset + signature->common_parameter_heap_offsets[i])});
    }
    if (signature->num_desc_samplers) {
      dev_ctx->get_device()->CopyDescriptorsSimple(
          signature->num_desc_samplers,
          {dev_ctx->get_sampler_desc_heap()->get_heap()->GetCPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_sampler_desc_heap()->get_element_size() * sampler_heap_offset},
          cpu_common_heap->GetCPUDescriptorHandleForHeapStart(),
          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      ito(signature->num_params_samplers) cmd->SetComputeRootDescriptorTable(
          signature->num_params_common + i,
          {dev_ctx->get_sampler_desc_heap()->get_heap()->GetGPUDescriptorHandleForHeapStart().ptr +
           dev_ctx->get_sampler_desc_heap()->get_element_size() *
               (sampler_heap_offset + signature->sampler_parameter_heap_offsets[i])});
    }
  }
  void flush_push_constants(ComPtr<ID3D12GraphicsCommandList> cmd) {
    cmd->SetComputeRoot32BitConstants(signature->num_params_total - 1,
                                      signature->push_constants_size / 4, push_constants_data, 0);
  }
  void push_constants(void const *data, size_t offset, size_t size) override {
    memcpy(push_constants_data + offset, data, size);
  }
  // u32  get_push_constants_parameter_id() { return num_params_total - 1; }
  // u32  get_push_constants_size() { return push_constants_size; }
  void bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                    size_t size) override {
    ID3D12Resource *res          = dev_ctx->get_resource(buf_id.id);
    u32             space_offset = signature->common_parameter_heap_offsets[space];
    u32             binding_offset =
        signature->space_descs[space].ranges[binding].OffsetInDescriptorsFromTableStart;
    D3D12_CONSTANT_BUFFER_VIEW_DESC desc{};
    ASSERT_DEBUG((offset & 0xffu) == 0);
    desc.BufferLocation = res->GetGPUVirtualAddress() + offset;
    desc.SizeInBytes    = (size + 0xffU) & ~0xffu;
    dev_ctx->get_device()->CreateConstantBufferView(
        &desc, {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
                dev_ctx->get_common_desc_heap()->get_element_size() *
                    ((u64)space_offset + (u64)binding_offset)});
  }
  void bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) override { TRAP; }
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
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
    desc.Buffer.CounterOffsetInBytes = 0;
    desc.Buffer.FirstElement         = offset / 4;
    desc.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_RAW;
    desc.Buffer.NumElements          = size / 4;
    desc.Buffer.StructureByteStride  = 0;
    desc.Format                      = DXGI_FORMAT_R32_TYPELESS;
    u32 space_offset                 = signature->common_parameter_heap_offsets[space];
    u32 binding_offset =
        signature->space_descs[space].ranges[binding].OffsetInDescriptorsFromTableStart;
    dev_ctx->get_device()->CreateUnorderedAccessView(
        res, NULL, &desc,
        {cpu_common_heap->GetCPUDescriptorHandleForHeapStart().ptr +
         dev_ctx->get_sampler_desc_heap()->get_element_size() *
             ((u64)space_offset + (u64)binding_offset)});
  }
  void bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                    rd::Image_Subresource const &range, rd::Format format) override {
    TRAP;
  }
  void bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                        rd::Image_Subresource const &range, rd::Format format) override {
    TRAP;
  }
  void release() override { delete this; }
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

  public:
  void release() {
    resource_state_tracker.iter_pairs([=](Resource_ID res, D3D12_RESOURCE_STATES state) {
      if (res.type == (u32)Resource_Type::BUFFER) {
        buffer_barrier(res, rd::Buffer_Access::GENERIC);
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
    TRAP;
  }
  void copy_image_to_buffer(Resource_ID buf_id, size_t buffer_offset, Resource_ID img_id,
                            rd::Image_Copy const &dst_info) override {
    TRAP;
  }
  void copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                   size_t dst_offset, u32 size) override {
    ID3D12Resource *src = dev_ctx->get_resource(src_buf_id.id);
    ID3D12Resource *dst = dev_ctx->get_resource(dst_buf_id.id);
    cmd->CopyBufferRegion(dst, dst_offset, src, src_offset, size);
  }
  // Synchronization
  void image_barrier(Resource_ID image_id, rd::Image_Access access) override { TRAP; }
  void buffer_barrier(Resource_ID buf_id, rd::Buffer_Access access) override {
    D3D12_RESOURCE_BARRIER bar{};
    bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    bar.Transition.pResource   = dev_ctx->get_resource(buf_id.id);
    bar.Transition.Subresource = 0;
    if (resource_state_tracker.contains(buf_id)) {
      bar.Transition.StateBefore = resource_state_tracker.get(buf_id);
    } else {
      bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
      resource_state_tracker.insert(buf_id, D3D12_RESOURCE_STATE_COMMON);
    }
    bar.Transition.StateAfter          = to_dx(access);
    resource_state_tracker.get(buf_id) = bar.Transition.StateAfter;
    if (bar.Transition.StateAfter == bar.Transition.StateBefore) {
      bar.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      bar.UAV.pResource = dev_ctx->get_resource(buf_id.id);
    }
    cmd->ResourceBarrier(1, &bar);
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