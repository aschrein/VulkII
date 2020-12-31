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

namespace {
static std::atomic<int> g_thread_counter;
static int              get_thread_id() {
  static thread_local int id = g_thread_counter++;
  return id;
};
} // namespace

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
  HANDLE                         sc_wait_obj = 0;
  u32                            cur_cmd_id  = 0;
  // ComPtr<ID3D12DescriptorHeap> dsv_desc_heap;
  ComPtr<ID3D12DescriptorHeap> sampler_desc_heap;
  ComPtr<ID3D12DescriptorHeap> common_desc_heap;
  std::mutex                   mutex;

  public:
  ComPtr<ID3D12Device2>             get_device() { return device; }
  ComPtr<ID3D12GraphicsCommandList> alloc_graphics_cmd() {
    ComPtr<ID3D12GraphicsCommandList> out;
    DX_ASSERT_OK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           cmd_allocs[cur_cmd_id].Get(), NULL, IID_PPV_ARGS(&out)));
    return out;
  }
  void bind_desc_heaps(ID3D12GraphicsCommandList *cmd) {
    ID3D12DescriptorHeap *descs[] = {
        common_desc_heap.Get(),
        sampler_desc_heap.Get(),
    };
    cmd->SetDescriptorHeaps(2, descs);
  }
  ComPtr<ID3D12DescriptorHeap> get_sampler_desc_heap() { return sampler_desc_heap; }
  ComPtr<ID3D12DescriptorHeap> get_common_desc_heap() { return common_desc_heap; }
  DX12Device(void *hdl) {
    {
      ComPtr<ID3D12Debug> debugInterface;
      DX_ASSERT_OK(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
      debugInterface->EnableDebugLayer();
    }
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_0;
    DX_ASSERT_OK(D3D12CreateDevice(NULL, featureLevel, IID_PPV_ARGS(&device)));

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
      D3D12_DESCRIPTOR_HEAP_DESC desc = {};
      desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
      desc.NumDescriptors             = 1 << 20;
      desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
      DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&common_desc_heap)));
    }
    {
      D3D12_DESCRIPTOR_HEAP_DESC desc = {};
      desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
      desc.NumDescriptors             = 1 << 20;
      desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
      DX_ASSERT_OK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&sampler_desc_heap)));
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
    TRAP;
  }
  Resource_ID create_shader(rd::Stage_t type, string_ref text,
                            Pair<string_ref, string_ref> *defines, size_t num_defines) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_sampler(rd::Sampler_Create_Info const &info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  void release_resource(Resource_ID id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_event() override {
    std::lock_guard<std::mutex> _lock(mutex);
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
    TRAP;
  }
  void unmap_buffer(Resource_ID id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_render_pass(rd::Render_Pass_Create_Info const &info) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_compute_pso(rd::IBinding_Table *table, Resource_ID cs) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  Resource_ID create_graphics_pso(rd::IBinding_Table *table, Resource_ID render_pass,
                                  rd::Graphics_Pipeline_State const &) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::IBinding_Table *create_binding_table(rd::Binding_Table_Create_Info const *infos,
                                           u32 num_tables, u32 push_constants_size) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::ICtx *start_render_pass(Resource_ID render_pass) override {
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
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  bool get_event_state(Resource_ID id) override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  rd::Impl_t getImplType() override { TRAP; }
  void       release() override {
    std::lock_guard<std::mutex> _lock(mutex);
    TRAP;
  }
  void start_frame() override { std::lock_guard<std::mutex> _lock(mutex); }
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

class DX12Binding_Table : public rd::IBinding_Table {
  DX12Device *                dev_ctx = NULL;
  ComPtr<ID3D12RootSignature> root_signature;

  public:
  static DX12Binding_Table *create(DX12Device *dev_ctx, rd::Binding_Table_Create_Info const *infos,
                                   u32 num_tables, u32 push_constants_size) {
    DX12Binding_Table *out                          = new DX12Binding_Table;
    out->dev_ctx                                    = dev_ctx;
    auto                                     device = dev_ctx->get_device();
    InlineArray<D3D12_ROOT_PARAMETER, 0x100> params{};

    u32 num_descs    = 0;
    u32 num_samplers = 0;
    ito(num_tables) {
      auto const &                               info = infos[i];
      InlineArray<D3D12_DESCRIPTOR_RANGE, 0x100> ranges{};
      jto(info.bindings.size) {
        auto                   b                     = info.bindings[j];
        D3D12_DESCRIPTOR_RANGE desc_range            = {};
        desc_range.RangeType                         = to_dx(b.type);
        desc_range.NumDescriptors                    = b.num_array_elems;
        desc_range.BaseShaderRegister                = b.binding;
        desc_range.RegisterSpace                     = i;
        desc_range.OffsetInDescriptorsFromTableStart = b.binding;
        num_descs = MAX(num_descs, b.binding + b.num_array_elems);
        ranges.push(desc_range);
      }
      D3D12_ROOT_PARAMETER param                = {};
      param.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      param.DescriptorTable.NumDescriptorRanges = ranges.size;
      param.DescriptorTable.pDescriptorRanges   = &ranges[0];
      param.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
      params.push(param);
    }
    {
      D3D12_ROOT_PARAMETER param{};

      param.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      param.Constants.ShaderRegister = 0;
      param.Constants.RegisterSpace  = 0;
      param.Constants.Num32BitValues = push_constants_size / 4;
      param.ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;
      params.push(param);
    }
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters             = params.size;
    desc.pParameters               = &params[0];
    desc.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ID3DBlob *blob                 = NULL;
    DX_ASSERT_OK(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL));
    DX_ASSERT_OK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                             IID_PPV_ARGS(&out->root_signature)));
    blob->Release();
  }
  void bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                    size_t size) override {
    TRAP;
  }
  void bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) override { TRAP; }
  void bind_UAV_buffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                       size_t size) override {
    TRAP;
  }
  void bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                    rd::Image_Subresource const &range, rd::Format format) override {
    TRAP;
  }
  void bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                        rd::Image_Subresource const &range, rd::Format format) override {
    TRAP;
  }
  void release() override { TRAP; }
  void clear_bindings() override { TRAP; }
};

class DX12Context : public rd::ICtx {
  ComPtr<ID3D12GraphicsCommandList> cmd;
  rd::Pass_t                        type;
  DX12Device *                      device = NULL;

  public:
  DX12Context(DX12Device *device, rd::Pass_t type) : type(type), device(device) {
    cmd = device->alloc_graphics_cmd();
  }
  void finalize() {}
  void bind_table(rd::IBinding_Table *table) override { //
    // cmd->SetDescriptorHeaps
     //cmd->SetGraphicsRootDescriptorTable
    TRAP;
  }
  void push_constants(void const *data, size_t offset, size_t size) override { TRAP; }
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
  void bind_compute(Resource_ID id) override { TRAP; }
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) override { TRAP; }
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
    TRAP;
  }
  // Synchronization
  void image_barrier(Resource_ID image_id, u32 access_flags, rd::Image_Layout layout) override {
    TRAP;
  }
  void buffer_barrier(Resource_ID buf_id, u32 access_flags) override { TRAP; }
  void insert_event(Resource_ID id) override { TRAP; }
  void insert_timestamp(Resource_ID timestamp_id) override { TRAP; }
};

rd::ICtx *DX12Device::start_compute_pass() {
  std::lock_guard<std::mutex> _lock(mutex);
  return new DX12Context(this, rd::Pass_t::COMPUTE);
}
void DX12Device::end_compute_pass(rd::ICtx *ctx) {
  std::lock_guard<std::mutex> _lock(mutex);
  ((DX12Context *)ctx)->finalize();
}
} // namespace

namespace rd {
rd::IDevice *create_dx12(void *window_handler) { return new DX12Device(window_handler); }
} // namespace rd