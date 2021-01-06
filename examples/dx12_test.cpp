#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "script.hpp"
#include "utils.hpp"
#include <SDL.h>
#include <SDL_syswm.h>

#if 0

#  include <atomic>
#  include <imgui.h>
#  include <mutex>
#  include <thread>

class Event_Consumer : public IGUI_Pass {
  public:
  void consume(void *_event) override {
    SDL_Event *event = (SDL_Event *)_event;
    if (imgui_initialized) {
      ImGui_ImplSDL2_ProcessEvent(event);
    }
    if (event->type == SDL_MOUSEMOTION) {
      SDL_MouseMotionEvent *m = (SDL_MouseMotionEvent *)event;
    }
  }
  void init(rd::Pass_Mng *pmng) override {}
  void on_init(rd::IFactory *factory) override {
    Image2D *image = load_image(stref_s("images/ECurtis.png"));
    defer(if (image) image->release());
    ASSERT_ALWAYS(image);
    RenderDoc_CTX::start();
    Resource_ID texture =
        Mip_Builder::create_image(factory, image, (u32)rd::Image_Usage_Bits::USAGE_SAMPLED);
    Resource_ID rw_texture{};
    {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA32_FLOAT;
      info.width      = image->width;
      info.height     = image->height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV;
      rw_texture      = factory->create_image(info);
    }
    defer(factory->release_resource(texture));
    rd::Imm_Ctx *ctx = factory->start_compute_pass();
    ctx->CS_set_shader(factory->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] Texture2D<float4>   rtex          : register(t0, space0);
[[vk::binding(1, 0)]] RWTexture2D<float4> rwtex         : register(u1, space0);
[[vk::binding(2, 0)]] SamplerState        ss            : register(s2, space0);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  uint width, height;
  rwtex.GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height)
    return;
  float2 uv = (float2(tid.xy) + float2(0.5f, 0.5f)) / float2(width, height);
  rwtex[tid.xy] = rtex.SampleLevel(ss, uv, 0.0f);
}
)"),
                                                  NULL, 0));
    ctx->image_barrier(texture, (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_ONLY_OPTIMAL);
    ctx->image_barrier(rw_texture, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_image(0, 0, 0, texture, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    ctx->bind_rw_image(0, 1, 0, rw_texture, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    Resource_ID sampler_state{};
    {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
      info.max_anisotropy = 16.0f;
      sampler_state       = factory->create_sampler(info);
    }
    defer(factory->release_resource(sampler_state));
    ctx->bind_sampler(0, 2, sampler_state);
    ctx->dispatch(image->width / 16 + 1, image->height / 16 + 1, 1);
    ctx->CS_set_shader(factory->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    BufferOut.Store(DTid.x * 4, DTid.x);
}
)"),
                                                  NULL, 0));
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
    buf_info.size       = sizeof(u32) * 1024;
    Resource_ID res_id  = factory->create_buffer(buf_info);
    ctx->bind_storage_buffer(0, 0, res_id, 0, buf_info.size);
    ctx->bind_storage_buffer(0, 1, res_id, 0, buf_info.size);
    ctx->dispatch(1, 1, 1);
    ctx->CS_set_shader(factory->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> BufferIn : register(u1, space0);

struct CullPushConstants
{
  uint val;
};
[[vk::push_constant]] ConstantBuffer<CullPushConstants> pc : register(b0, space0);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    BufferOut.Store(DTid.x * 4, BufferIn.Load(DTid.x) * pc.val);
}
)"),
                                                  NULL, 0));
    u32 val = 3;
    ctx->push_constants(&val, 0, 4);
    ctx->dispatch(1, 1, 1);
    Resource_ID event_id = ctx->insert_event();
    factory->end_compute_pass(ctx);
    while (!factory->get_event_state(event_id)) fprintf(stdout, "waiting...\n");
    factory->release_resource(event_id);
    u32 *map = (u32 *)factory->map_buffer(res_id);
    ito(1024) fprintf(stdout, "%i ", map[i]);
    factory->unmap_buffer(res_id);
    factory->release_resource(res_id);
    RenderDoc_CTX::end();
  }
  void on_release(rd::IFactory *factory) override {}
  void on_frame(rd::IFactory *factory) override {}
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  TMP_STORAGE_SCOPE;

  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
  SDL_Window *window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        512, 512, SDL_WINDOW_RESIZABLE);
  defer({
    if (window) {
      SDL_DestroyWindow(window);
    }
    SDL_Quit();
  });

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  SDL_GetWindowWMInfo(window, &wmInfo);
  HWND hwnd = wmInfo.info.win.window;

  rd::IFactory *factory = rd::create_vulkan((void *)&hwnd);

  while (true) {
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
  }

  IGUI_Pass *gui = new Event_Consumer;
  /*gui->init(pmng);
  pmng->set_event_consumer(gui);
  std::thread dx12_thread = std::thread([=] { pmng->loop(); });
  dx12_thread.join();

  rd::Pass_Mng *pmng = rd::create_dx12_pass_mng();
  IGUI_Pass *   gui  = new Event_Consumer;
  gui->init(pmng);
  pmng->set_event_consumer(gui);
  std::thread dx12_thread = std::thread([=] { pmng->loop(); });
  dx12_thread.join();*/
  return 0;
}

#else

// dear imgui: standalone example application for DirectX 12
// If you are new to dear imgui, see examples/README.txt and documentation at the top of imgui.cpp.
// FIXME: 64-bit only for now! (Because sizeof(ImTextureId) == sizeof(void*))
#  define WIN32_LEAN_AND_MEAN
#  include <DirectXMath.h>
#  include <Windows.h>
#  include <d3d12.h>
#  include <d3dcompiler.h>
#  include <dxgi1_6.h>
#  include <wrl.h>
using namespace Microsoft::WRL;

#  define DX_ASSERT_OK(x)                                                                          \
    do {                                                                                           \
      HRESULT __res = x;                                                                           \
      if (FAILED(__res)) {                                                                         \
        fprintf(stderr, "__res: %i\n", (i32)__res);                                                \
        TRAP;                                                                                      \
      }                                                                                            \
    } while (0)
#  include "3rdparty/imgui/examples/imgui_impl_dx12.h"
#  include "3rdparty/imgui/examples/imgui_impl_win32.h"
#  include "imgui.h"
#  include <d3d12.h>
#  include <dxgi1_4.h>
#  include <tchar.h>

#  define _DEBUG

#  ifdef _DEBUG
#    define DX12_ENABLE_DEBUG_LAYER
#  endif

#  ifdef DX12_ENABLE_DEBUG_LAYER
#    include <dxgidebug.h>
#    pragma comment(lib, "dxguid.lib")
#  endif

struct FrameContext {
  ID3D12CommandAllocator *CommandAllocator;
  UINT64                  FenceValue;
};

// Data
static int const    NUM_FRAMES_IN_FLIGHT                 = 3;
static FrameContext g_frameContext[NUM_FRAMES_IN_FLIGHT] = {};
static UINT         g_frameIndex                         = 0;

static int const                   NUM_BACK_BUFFERS                               = 3;
static ID3D12Device *              g_pd3dDevice                                   = NULL;
static ID3D12DescriptorHeap *      g_pd3dRtvDescHeap                              = NULL;
static ID3D12DescriptorHeap *      g_pd3dSrvDescHeap                              = NULL;
static ID3D12CommandQueue *        g_pd3dCommandQueue                             = NULL;
static ID3D12GraphicsCommandList * g_pd3dCommandList                              = NULL;
static ID3D12Fence *               g_fence                                        = NULL;
static HANDLE                      g_fenceEvent                                   = NULL;
static UINT64                      g_fenceLastSignaledValue                       = 0;
static IDXGISwapChain3 *           g_pSwapChain                                   = NULL;
static HANDLE                      g_hSwapChainWaitableObject                     = NULL;
static ID3D12Resource *            g_mainRenderTargetResource[NUM_BACK_BUFFERS]   = {};
static D3D12_CPU_DESCRIPTOR_HANDLE g_mainRenderTargetDescriptor[NUM_BACK_BUFFERS] = {};

// Forward declarations of helper functions
bool          CreateDeviceD3D(HWND hWnd);
void          CleanupDeviceD3D();
void          CreateRenderTarget();
void          CleanupRenderTarget();
void          WaitForLastSubmittedFrame();
FrameContext *WaitForNextFrameResources();
void          ResizeSwapChain(HWND hWnd, int width, int height);
// LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Main code
int main(int, char **) {

  // Create application window
  // ImGui_ImplWin32_EnableDpiAwareness();
  TMP_STORAGE_SCOPE;

  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
  SDL_Window *window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                        512, 512, SDL_WINDOW_RESIZABLE);
  defer({
    if (window) {
      SDL_DestroyWindow(window);
    }
    SDL_Quit();
  });
  while (true) {
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
  }

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  SDL_GetWindowWMInfo(window, &wmInfo);
  HWND hwnd = wmInfo.info.win.window;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  // Initialize Direct3D
  if (!CreateDeviceD3D(hwnd)) {
    CleanupDeviceD3D();
    //::UnregisterClass(wc.lpszClassName, wc.hInstance);
    return 1;
  }

  // Setup Dear ImGui context

  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;   // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
  // io.ConfigViewportsNoAutoMerge = true;
  // io.ConfigViewportsNoTaskBarIcon = true;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsClassic();

  // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look
  // identical to regular ones.
  ImGuiStyle &style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding              = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  // Setup Platform/Renderer bindings
  ImGui_ImplWin32_Init(hwnd);
  ImGui_ImplDX12_Init(g_pd3dDevice, NUM_FRAMES_IN_FLIGHT, DXGI_FORMAT_R8G8B8A8_UNORM,
                      g_pd3dSrvDescHeap, g_pd3dSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
                      g_pd3dSrvDescHeap->GetGPUDescriptorHandleForHeapStart());

  // Show the window
  ::ShowWindow(hwnd, SW_SHOWDEFAULT);
  ::UpdateWindow(hwnd);

  // Load Fonts
  // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple
  // fonts and use ImGui::PushFont()/PopFont() to select them.
  // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the
  // font among multiple.
  // - If the file cannot be loaded, the function will return NULL. Please handle those errors in
  // your application (e.g. use an assertion, or display an error and quit).
  // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when
  // calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
  // - Read 'docs/FONTS.md' for more instructions and details.
  // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to
  // write a double backslash \\ !
  // io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
  // ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL,
  // io.Fonts->GetGlyphRangesJapanese()); IM_ASSERT(font != NULL);

  // Our state
  bool   show_demo_window    = true;
  bool   show_another_window = false;
  ImVec4 clear_color         = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  // Main loop
  MSG msg;
  ZeroMemory(&msg, sizeof(msg));
  while (msg.message != WM_QUIT) {
    // Poll and handle messages (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui
    // wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main
    // application.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main
    // application. Generally you may always pass all inputs to dear imgui, and hide them from your
    // application based on those two flags.
    if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
      ::TranslateMessage(&msg);
      ::DispatchMessage(&msg);
      continue;
    }

    // Start the Dear ImGui frame
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can
    // browse its code to learn more about Dear ImGui!).
    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named
    // window.
    {
      static float f       = 0.0f;
      static int   counter = 0;

      ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

      ImGui::Text(
          "This is some useful text."); // Display some text (you can use a format strings too)
      ImGui::Checkbox("Demo Window",
                      &show_demo_window); // Edit bools storing our window open/close state
      ImGui::Checkbox("Another Window", &show_another_window);

      ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
      ImGui::ColorEdit3("clear color", (float *)&clear_color); // Edit 3 floats representing a color

      if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true
                                   // when edited/activated)
        counter++;
      ImGui::SameLine();
      ImGui::Text("counter = %d", counter);

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
    }

    // 3. Show another simple window.
    if (show_another_window) {
      ImGui::Begin(
          "Another Window",
          &show_another_window); // Pass a pointer to our bool variable (the window will have a
                                 // closing button that will clear the bool when clicked)
      ImGui::Text("Hello from another window!");
      if (ImGui::Button("Close Me")) show_another_window = false;
      ImGui::End();
    }

    // Rendering
    FrameContext *frameCtxt     = WaitForNextFrameResources();
    UINT          backBufferIdx = g_pSwapChain->GetCurrentBackBufferIndex();
    frameCtxt->CommandAllocator->Reset();

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource   = g_mainRenderTargetResource[backBufferIdx];
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;

    g_pd3dCommandList->Reset(frameCtxt->CommandAllocator, NULL);
    g_pd3dCommandList->ResourceBarrier(1, &barrier);
    g_pd3dCommandList->ClearRenderTargetView(g_mainRenderTargetDescriptor[backBufferIdx],
                                             (float *)&clear_color, 0, NULL);
    g_pd3dCommandList->OMSetRenderTargets(1, &g_mainRenderTargetDescriptor[backBufferIdx], FALSE,
                                          NULL);
    g_pd3dCommandList->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);
    ImGui::Render();
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_pd3dCommandList);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
    g_pd3dCommandList->ResourceBarrier(1, &barrier);
    g_pd3dCommandList->Close();

    g_pd3dCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList *const *)&g_pd3dCommandList);

    // Update and Render additional Platform Windows
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault(NULL, (void *)g_pd3dCommandList);
    }

    g_pSwapChain->Present(1, 0); // Present with vsync
    // g_pSwapChain->Present(0, 0); // Present without vsync

    UINT64 fenceValue = g_fenceLastSignaledValue + 1;
    g_pd3dCommandQueue->Signal(g_fence, fenceValue);
    g_fenceLastSignaledValue = fenceValue;
    frameCtxt->FenceValue    = fenceValue;
  }

  WaitForLastSubmittedFrame();
  ImGui_ImplDX12_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();

  CleanupDeviceD3D();
  //::DestroyWindow(hwnd);
  //::UnregisterClass(wc.lpszClassName, wc.hInstance);

  return 0;
}

// Helper functions

bool CreateDeviceD3D(HWND hWnd) {
  // Setup swap chain
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

#  ifdef DX12_ENABLE_DEBUG_LAYER
  ID3D12Debug *pdx12Debug = NULL;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pdx12Debug)))) {
    pdx12Debug->EnableDebugLayer();
    pdx12Debug->Release();
  }
#  endif

  D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_12_1;
  if (D3D12CreateDevice(NULL, featureLevel, IID_PPV_ARGS(&g_pd3dDevice)) != S_OK) return false;

  {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    desc.NumDescriptors             = NUM_BACK_BUFFERS;
    desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    desc.NodeMask                   = 1;
    if (g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dRtvDescHeap)) != S_OK)
      return false;

    SIZE_T rtvDescriptorSize =
        g_pd3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = g_pd3dRtvDescHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < NUM_BACK_BUFFERS; i++) {
      g_mainRenderTargetDescriptor[i] = rtvHandle;
      rtvHandle.ptr += rtvDescriptorSize;
    }
  }

  {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors             = 1;
    desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dSrvDescHeap)) != S_OK)
      return false;
  }

  {
    D3D12_COMMAND_QUEUE_DESC desc = {};
    desc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    desc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.NodeMask                 = 1;
    if (g_pd3dDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&g_pd3dCommandQueue)) != S_OK)
      return false;
  }

  for (UINT i = 0; i < NUM_FRAMES_IN_FLIGHT; i++)
    if (g_pd3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&g_frameContext[i].CommandAllocator)) !=
        S_OK)
      return false;

  if (g_pd3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      g_frameContext[0].CommandAllocator, NULL,
                                      IID_PPV_ARGS(&g_pd3dCommandList)) != S_OK ||
      g_pd3dCommandList->Close() != S_OK)
    return false;

  if (g_pd3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)) != S_OK)
    return false;

  g_fenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  if (g_fenceEvent == NULL) return false;

  {
    IDXGIFactory4 *  dxgiFactory = NULL;
    IDXGISwapChain1 *swapChain1  = NULL;
    if (CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)) != S_OK ||
        dxgiFactory->CreateSwapChainForHwnd(g_pd3dCommandQueue, hWnd, &sd, NULL, NULL,
                                            &swapChain1) != S_OK ||
        swapChain1->QueryInterface(IID_PPV_ARGS(&g_pSwapChain)) != S_OK)
      return false;
    swapChain1->Release();
    dxgiFactory->Release();
    g_pSwapChain->SetMaximumFrameLatency(NUM_BACK_BUFFERS);
    g_hSwapChainWaitableObject = g_pSwapChain->GetFrameLatencyWaitableObject();
  }

  CreateRenderTarget();
  return true;
}

void CleanupDeviceD3D() {
  CleanupRenderTarget();
  if (g_pSwapChain) {
    g_pSwapChain->Release();
    g_pSwapChain = NULL;
  }
  if (g_hSwapChainWaitableObject != NULL) {
    CloseHandle(g_hSwapChainWaitableObject);
  }
  for (UINT i = 0; i < NUM_FRAMES_IN_FLIGHT; i++)
    if (g_frameContext[i].CommandAllocator) {
      g_frameContext[i].CommandAllocator->Release();
      g_frameContext[i].CommandAllocator = NULL;
    }
  if (g_pd3dCommandQueue) {
    g_pd3dCommandQueue->Release();
    g_pd3dCommandQueue = NULL;
  }
  if (g_pd3dCommandList) {
    g_pd3dCommandList->Release();
    g_pd3dCommandList = NULL;
  }
  if (g_pd3dRtvDescHeap) {
    g_pd3dRtvDescHeap->Release();
    g_pd3dRtvDescHeap = NULL;
  }
  if (g_pd3dSrvDescHeap) {
    g_pd3dSrvDescHeap->Release();
    g_pd3dSrvDescHeap = NULL;
  }
  if (g_fence) {
    g_fence->Release();
    g_fence = NULL;
  }
  if (g_fenceEvent) {
    CloseHandle(g_fenceEvent);
    g_fenceEvent = NULL;
  }
  if (g_pd3dDevice) {
    g_pd3dDevice->Release();
    g_pd3dDevice = NULL;
  }

#  ifdef DX12_ENABLE_DEBUG_LAYER
  IDXGIDebug1 *pDebug = NULL;
  if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&pDebug)))) {
    pDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_SUMMARY);
    pDebug->Release();
  }
#  endif
}

void CreateRenderTarget() {
  for (UINT i = 0; i < NUM_BACK_BUFFERS; i++) {
    ID3D12Resource *pBackBuffer = NULL;
    g_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, g_mainRenderTargetDescriptor[i]);
    g_mainRenderTargetResource[i] = pBackBuffer;
  }
}

void CleanupRenderTarget() {
  WaitForLastSubmittedFrame();

  for (UINT i = 0; i < NUM_BACK_BUFFERS; i++)
    if (g_mainRenderTargetResource[i]) {
      g_mainRenderTargetResource[i]->Release();
      g_mainRenderTargetResource[i] = NULL;
    }
}

void WaitForLastSubmittedFrame() {
  FrameContext *frameCtxt = &g_frameContext[g_frameIndex % NUM_FRAMES_IN_FLIGHT];

  UINT64 fenceValue = frameCtxt->FenceValue;
  if (fenceValue == 0) return; // No fence was signaled

  frameCtxt->FenceValue = 0;
  if (g_fence->GetCompletedValue() >= fenceValue) return;

  g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent);
  WaitForSingleObject(g_fenceEvent, INFINITE);
}

FrameContext *WaitForNextFrameResources() {
  UINT nextFrameIndex = g_frameIndex + 1;
  g_frameIndex        = nextFrameIndex;

  HANDLE waitableObjects[]  = {g_hSwapChainWaitableObject, NULL};
  DWORD  numWaitableObjects = 1;

  FrameContext *frameCtxt  = &g_frameContext[nextFrameIndex % NUM_FRAMES_IN_FLIGHT];
  UINT64        fenceValue = frameCtxt->FenceValue;
  if (fenceValue != 0) // means no fence was signaled
  {
    frameCtxt->FenceValue = 0;
    g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent);
    waitableObjects[1] = g_fenceEvent;
    numWaitableObjects = 2;
  }

  WaitForMultipleObjects(numWaitableObjects, waitableObjects, TRUE, INFINITE);

  return frameCtxt;
}

void ResizeSwapChain(HWND hWnd, int width, int height) {
  DXGI_SWAP_CHAIN_DESC1 sd;
  g_pSwapChain->GetDesc1(&sd);
  sd.Width  = width;
  sd.Height = height;

  IDXGIFactory4 *dxgiFactory = NULL;
  g_pSwapChain->GetParent(IID_PPV_ARGS(&dxgiFactory));

  g_pSwapChain->Release();
  CloseHandle(g_hSwapChainWaitableObject);

  IDXGISwapChain1 *swapChain1 = NULL;
  dxgiFactory->CreateSwapChainForHwnd(g_pd3dCommandQueue, hWnd, &sd, NULL, NULL, &swapChain1);
  swapChain1->QueryInterface(IID_PPV_ARGS(&g_pSwapChain));
  swapChain1->Release();
  dxgiFactory->Release();

  g_pSwapChain->SetMaximumFrameLatency(NUM_BACK_BUFFERS);

  g_hSwapChainWaitableObject = g_pSwapChain->GetFrameLatencyWaitableObject();
  assert(g_hSwapChainWaitableObject != NULL);
}

//// Forward declare message handler from imgui_impl_win32.cpp
// extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam,
//                                                             LPARAM lParam);

// Win32 message handler
// LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
//  if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
//
//  switch (msg) {
//  case WM_SIZE:
//    if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED) {
//      WaitForLastSubmittedFrame();
//      ImGui_ImplDX12_InvalidateDeviceObjects();
//      CleanupRenderTarget();
//      ResizeSwapChain(hWnd, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam));
//      CreateRenderTarget();
//      ImGui_ImplDX12_CreateDeviceObjects();
//    }
//    return 0;
//  case WM_SYSCOMMAND:
//    if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
//      return 0;
//    break;
//  case WM_DESTROY: ::PostQuitMessage(0); return 0;
//  }
//  return ::DefWindowProc(hWnd, msg, wParam, lParam);
//}

// dear imgui: Renderer for DirectX12
// This needs to be used along with a Platform Binding (e.g. Win32)

// Implemented features:
//  [X] Renderer: User texture binding. Use 'D3D12_GPU_DESCRIPTOR_HANDLE' as ImTextureID. Read the
//  FAQ about ImTextureID! [X] Renderer: Multi-viewport support. Enable with 'io.ConfigFlags |=
//  ImGuiConfigFlags_ViewportsEnable'.
//      FIXME: The transition from removing a viewport and moving the window in an existing hosted
//      viewport tends to flicker.
//  [X] Renderer: Support for large meshes (64k+ vertices) with 16-bit indices.
// Missing features, issues:
//  [ ] 64-bit only for now! (Because sizeof(ImTextureId) == sizeof(void*)). See
//  github.com/ocornut/imgui/pull/301

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example
// of using this. If you are new to dear imgui, read examples/README.txt and read the documentation
// at the top of imgui.cpp. https://github.com/ocornut/imgui

// CHANGELOG
// (minor and older changes stripped away, please see git history for details)
//  2020-XX-XX: Platform: Added support for multiple windows via the ImGuiPlatformIO interface.
//  2019-10-18: DirectX12: *BREAKING CHANGE* Added extra ID3D12DescriptorHeap parameter to
//  ImGui_ImplDX12_Init() function. 2019-05-29: DirectX12: Added support for large mesh (64K+
//  vertices), enable ImGuiBackendFlags_RendererHasVtxOffset flag. 2019-04-30: DirectX12: Added
//  support for special ImDrawCallback_ResetRenderState callback to reset render state. 2019-03-29:
//  Misc: Various minor tidying up. 2018-12-03: Misc: Added #pragma comment statement to
//  automatically link with d3dcompiler.lib when using D3DCompile(). 2018-11-30: Misc: Setting up
//  io.BackendRendererName so it can be displayed in the About Window. 2018-06-12: DirectX12: Moved
//  the ID3D12GraphicsCommandList* parameter from NewFrame() to RenderDrawData(). 2018-06-08: Misc:
//  Extracted imgui_impl_dx12.cpp/.h away from the old combined DX12+Win32 example. 2018-06-08:
//  DirectX12: Use draw_data->DisplayPos and draw_data->DisplaySize to setup projection matrix and
//  clipping rectangle (to ease support for future multi-viewport). 2018-02-22: Merged into master
//  with all Win32 code synchronized to other examples.

#  include "imgui.h"

// DirectX
#  include <d3d12.h>
#  include <d3dcompiler.h>
#  include <dxgi1_4.h>
#  ifdef _MSC_VER
#    pragma comment(lib, "d3dcompiler") // Automatically link with d3dcompiler.lib as we are using
                                        // D3DCompile() below.
#  endif

// DirectX data
// static ID3D12Device*                g_pd3dDevice = NULL;
static ID3D12RootSignature *       g_pRootSignature        = NULL;
static ID3D12PipelineState *       g_pPipelineState        = NULL;
static DXGI_FORMAT                 g_RTVFormat             = DXGI_FORMAT_UNKNOWN;
static ID3D12Resource *            g_pFontTextureResource  = NULL;
static D3D12_CPU_DESCRIPTOR_HANDLE g_hFontSrvCpuDescHandle = {};
static D3D12_GPU_DESCRIPTOR_HANDLE g_hFontSrvGpuDescHandle = {};
// static ID3D12DescriptorHeap*        g_pd3dSrvDescHeap = NULL;
static UINT g_numFramesInFlight = 0;

// Buffers used during the rendering of a frame
struct ImGui_ImplDX12_RenderBuffers {
  ID3D12Resource *IndexBuffer;
  ID3D12Resource *VertexBuffer;
  int             IndexBufferSize;
  int             VertexBufferSize;
};

// Buffers used for secondary viewports created by the multi-viewports systems
struct ImGui_ImplDX12_FrameContext {
  ID3D12CommandAllocator *    CommandAllocator;
  ID3D12Resource *            RenderTarget;
  D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetCpuDescriptors;
};

// Helper structure we store in the void* RendererUserData field of each ImGuiViewport to easily
// retrieve our backend data. Main viewport created by application will only use the Resources
// field. Secondary viewports created by this back-end will use all the fields (including Window
// fields),
struct ImGuiViewportDataDx12 {
  // Window
  ID3D12CommandQueue *         CommandQueue;
  ID3D12GraphicsCommandList *  CommandList;
  ID3D12DescriptorHeap *       RtvDescHeap;
  IDXGISwapChain3 *            SwapChain;
  ID3D12Fence *                Fence;
  UINT64                       FenceSignaledValue;
  HANDLE                       FenceEvent;
  ImGui_ImplDX12_FrameContext *FrameCtx;

  // Render buffers
  UINT                          FrameIndex;
  ImGui_ImplDX12_RenderBuffers *FrameRenderBuffers;

  ImGuiViewportDataDx12() {
    CommandQueue       = NULL;
    CommandList        = NULL;
    RtvDescHeap        = NULL;
    SwapChain          = NULL;
    Fence              = NULL;
    FenceSignaledValue = 0;
    FenceEvent         = NULL;
    FrameCtx           = new ImGui_ImplDX12_FrameContext[g_numFramesInFlight];
    FrameIndex         = UINT_MAX;
    FrameRenderBuffers = new ImGui_ImplDX12_RenderBuffers[g_numFramesInFlight];

    for (UINT i = 0; i < g_numFramesInFlight; ++i) {
      FrameCtx[i].CommandAllocator = NULL;
      FrameCtx[i].RenderTarget     = NULL;

      // Create buffers with a default size (they will later be grown as needed)
      FrameRenderBuffers[i].IndexBuffer      = NULL;
      FrameRenderBuffers[i].VertexBuffer     = NULL;
      FrameRenderBuffers[i].VertexBufferSize = 5000;
      FrameRenderBuffers[i].IndexBufferSize  = 10000;
    }
  }
  ~ImGuiViewportDataDx12() {
    IM_ASSERT(CommandQueue == NULL && CommandList == NULL);
    IM_ASSERT(RtvDescHeap == NULL);
    IM_ASSERT(SwapChain == NULL);
    IM_ASSERT(Fence == NULL);
    IM_ASSERT(FenceEvent == NULL);

    for (UINT i = 0; i < g_numFramesInFlight; ++i) {
      IM_ASSERT(FrameCtx[i].CommandAllocator == NULL && FrameCtx[i].RenderTarget == NULL);
      IM_ASSERT(FrameRenderBuffers[i].IndexBuffer == NULL &&
                FrameRenderBuffers[i].VertexBuffer == NULL);
    }

    delete[] FrameCtx;
    FrameCtx = NULL;
    delete[] FrameRenderBuffers;
    FrameRenderBuffers = NULL;
  }
};

template <typename T> static void SafeRelease(T *&res) {
  if (res) res->Release();
  res = NULL;
}

static void ImGui_ImplDX12_DestroyRenderBuffers(ImGui_ImplDX12_RenderBuffers *render_buffers) {
  SafeRelease(render_buffers->IndexBuffer);
  SafeRelease(render_buffers->VertexBuffer);
  render_buffers->IndexBufferSize = render_buffers->VertexBufferSize = 0;
}

struct VERTEX_CONSTANT_BUFFER {
  float mvp[4][4];
};

// Forward Declarations
static void ImGui_ImplDX12_InitPlatformInterface();
static void ImGui_ImplDX12_ShutdownPlatformInterface();

static void ImGui_ImplDX12_SetupRenderState(ImDrawData *draw_data, ID3D12GraphicsCommandList *ctx,
                                            ImGui_ImplDX12_RenderBuffers *fr) {
  // Setup orthographic projection matrix into our constant buffer
  // Our visible imgui space lies from draw_data->DisplayPos (top left) to
  // draw_data->DisplayPos+data_data->DisplaySize (bottom right).
  VERTEX_CONSTANT_BUFFER vertex_constant_buffer;
  {
    float L         = draw_data->DisplayPos.x;
    float R         = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T         = draw_data->DisplayPos.y;
    float B         = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    float mvp[4][4] = {
        {2.0f / (R - L), 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f / (T - B), 0.0f, 0.0f},
        {0.0f, 0.0f, 0.5f, 0.0f},
        {(R + L) / (L - R), (T + B) / (B - T), 0.5f, 1.0f},
    };
    memcpy(&vertex_constant_buffer.mvp, mvp, sizeof(mvp));
  }

  // Setup viewport
  D3D12_VIEWPORT vp;
  memset(&vp, 0, sizeof(D3D12_VIEWPORT));
  vp.Width    = draw_data->DisplaySize.x;
  vp.Height   = draw_data->DisplaySize.y;
  vp.MinDepth = 0.0f;
  vp.MaxDepth = 1.0f;
  vp.TopLeftX = vp.TopLeftY = 0.0f;
  ctx->RSSetViewports(1, &vp);

  // Bind shader and vertex buffers
  unsigned int             stride = sizeof(ImDrawVert);
  unsigned int             offset = 0;
  D3D12_VERTEX_BUFFER_VIEW vbv;
  memset(&vbv, 0, sizeof(D3D12_VERTEX_BUFFER_VIEW));
  vbv.BufferLocation = fr->VertexBuffer->GetGPUVirtualAddress() + offset;
  vbv.SizeInBytes    = fr->VertexBufferSize * stride;
  vbv.StrideInBytes  = stride;
  ctx->IASetVertexBuffers(0, 1, &vbv);
  D3D12_INDEX_BUFFER_VIEW ibv;
  memset(&ibv, 0, sizeof(D3D12_INDEX_BUFFER_VIEW));
  ibv.BufferLocation = fr->IndexBuffer->GetGPUVirtualAddress();
  ibv.SizeInBytes    = fr->IndexBufferSize * sizeof(ImDrawIdx);
  ibv.Format         = sizeof(ImDrawIdx) == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
  ctx->IASetIndexBuffer(&ibv);
  ctx->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  ctx->SetPipelineState(g_pPipelineState);
  ctx->SetGraphicsRootSignature(g_pRootSignature);
  ctx->SetGraphicsRoot32BitConstants(0, 16, &vertex_constant_buffer, 0);

  // Setup blend factor
  const float blend_factor[4] = {0.f, 0.f, 0.f, 0.f};
  ctx->OMSetBlendFactor(blend_factor);
}

// Render function
// (this used to be set in io.RenderDrawListsFn and called by ImGui::Render(), but you can now call
// this directly from your main loop)
void ImGui_ImplDX12_RenderDrawData(ImDrawData *draw_data, ID3D12GraphicsCommandList *ctx) {
  // Avoid rendering when minimized
  if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f) return;

  ImGuiViewportDataDx12 *render_data =
      (ImGuiViewportDataDx12 *)draw_data->OwnerViewport->RendererUserData;
  render_data->FrameIndex++;
  ImGui_ImplDX12_RenderBuffers *fr =
      &render_data->FrameRenderBuffers[render_data->FrameIndex % g_numFramesInFlight];

  // Create and grow vertex/index buffers if needed
  if (fr->VertexBuffer == NULL || fr->VertexBufferSize < draw_data->TotalVtxCount) {
    SafeRelease(fr->VertexBuffer);
    fr->VertexBufferSize = draw_data->TotalVtxCount + 5000;
    D3D12_HEAP_PROPERTIES props;
    memset(&props, 0, sizeof(D3D12_HEAP_PROPERTIES));
    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    D3D12_RESOURCE_DESC desc;
    memset(&desc, 0, sizeof(D3D12_RESOURCE_DESC));
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = fr->VertexBufferSize * sizeof(ImDrawVert);
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.Format           = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;
    if (g_pd3dDevice->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                                              IID_PPV_ARGS(&fr->VertexBuffer)) < 0)
      return;
  }
  if (fr->IndexBuffer == NULL || fr->IndexBufferSize < draw_data->TotalIdxCount) {
    SafeRelease(fr->IndexBuffer);
    fr->IndexBufferSize = draw_data->TotalIdxCount + 10000;
    D3D12_HEAP_PROPERTIES props;
    memset(&props, 0, sizeof(D3D12_HEAP_PROPERTIES));
    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    D3D12_RESOURCE_DESC desc;
    memset(&desc, 0, sizeof(D3D12_RESOURCE_DESC));
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = fr->IndexBufferSize * sizeof(ImDrawIdx);
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.Format           = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;
    if (g_pd3dDevice->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                                              IID_PPV_ARGS(&fr->IndexBuffer)) < 0)
      return;
  }

  // Upload vertex/index data into a single contiguous GPU buffer
  void *      vtx_resource, *idx_resource;
  D3D12_RANGE range;
  memset(&range, 0, sizeof(D3D12_RANGE));
  if (fr->VertexBuffer->Map(0, &range, &vtx_resource) != S_OK) return;
  if (fr->IndexBuffer->Map(0, &range, &idx_resource) != S_OK) return;
  ImDrawVert *vtx_dst = (ImDrawVert *)vtx_resource;
  ImDrawIdx * idx_dst = (ImDrawIdx *)idx_resource;
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList *cmd_list = draw_data->CmdLists[n];
    memcpy(vtx_dst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
    vtx_dst += cmd_list->VtxBuffer.Size;
    idx_dst += cmd_list->IdxBuffer.Size;
  }
  fr->VertexBuffer->Unmap(0, &range);
  fr->IndexBuffer->Unmap(0, &range);

  // Setup desired DX state
  ImGui_ImplDX12_SetupRenderState(draw_data, ctx, fr);

  // Render command lists
  // (Because we merged all buffers into a single one, we maintain our own offset into them)
  int    global_vtx_offset = 0;
  int    global_idx_offset = 0;
  ImVec2 clip_off          = draw_data->DisplayPos;
  for (int n = 0; n < draw_data->CmdListsCount; n++) {
    const ImDrawList *cmd_list = draw_data->CmdLists[n];
    for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
      const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[cmd_i];
      if (pcmd->UserCallback != NULL) {
        // User callback, registered via ImDrawList::AddCallback()
        // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request
        // the renderer to reset render state.)
        if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
          ImGui_ImplDX12_SetupRenderState(draw_data, ctx, fr);
        else
          pcmd->UserCallback(cmd_list, pcmd);
      } else {
        // Apply Scissor, Bind texture, Draw
        const D3D12_RECT r = {
            (LONG)(pcmd->ClipRect.x - clip_off.x), (LONG)(pcmd->ClipRect.y - clip_off.y),
            (LONG)(pcmd->ClipRect.z - clip_off.x), (LONG)(pcmd->ClipRect.w - clip_off.y)};
        ctx->SetGraphicsRootDescriptorTable(1, *(D3D12_GPU_DESCRIPTOR_HANDLE *)&pcmd->TextureId);
        ctx->RSSetScissorRects(1, &r);
        ctx->DrawIndexedInstanced(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset,
                                  pcmd->VtxOffset + global_vtx_offset, 0);
      }
    }
    global_idx_offset += cmd_list->IdxBuffer.Size;
    global_vtx_offset += cmd_list->VtxBuffer.Size;
  }
}

static void ImGui_ImplDX12_CreateFontsTexture() {
  // Build texture atlas
  ImGuiIO &      io = ImGui::GetIO();
  unsigned char *pixels;
  int            width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

  // Upload texture to graphics system
  {
    D3D12_HEAP_PROPERTIES props;
    memset(&props, 0, sizeof(D3D12_HEAP_PROPERTIES));
    props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment          = 0;
    desc.Width              = width;
    desc.Height             = height;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags              = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource *pTexture = NULL;
    DX_ASSERT_OK(g_pd3dDevice->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, NULL,
                                                       IID_PPV_ARGS(&pTexture)));

    UINT uploadPitch = (width * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u) &
                       ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u);
    UINT uploadSize         = height * uploadPitch;
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment          = 0;
    desc.Width              = uploadSize;
    desc.Height             = 1;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags              = D3D12_RESOURCE_FLAG_NONE;

    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    ID3D12Resource *uploadBuffer = NULL;
    HRESULT         hr = g_pd3dDevice->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, NULL,
                                                       IID_PPV_ARGS(&uploadBuffer));
    IM_ASSERT(SUCCEEDED(hr));

    void *      mapped = NULL;
    D3D12_RANGE range  = {0, uploadSize};
    hr                 = uploadBuffer->Map(0, &range, &mapped);
    IM_ASSERT(SUCCEEDED(hr));
    for (int y = 0; y < height; y++)
      memcpy((void *)((uintptr_t)mapped + y * uploadPitch), pixels + y * width * 4, width * 4);
    uploadBuffer->Unmap(0, &range);

    D3D12_TEXTURE_COPY_LOCATION srcLocation        = {};
    srcLocation.pResource                          = uploadBuffer;
    srcLocation.Type                               = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLocation.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    srcLocation.PlacedFootprint.Footprint.Width    = width;
    srcLocation.PlacedFootprint.Footprint.Height   = height;
    srcLocation.PlacedFootprint.Footprint.Depth    = 1;
    srcLocation.PlacedFootprint.Footprint.RowPitch = uploadPitch;

    D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
    dstLocation.pResource                   = pTexture;
    dstLocation.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLocation.SubresourceIndex            = 0;

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource   = pTexture;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    ID3D12Fence *fence = NULL;
    hr                 = g_pd3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    IM_ASSERT(SUCCEEDED(hr));

    HANDLE event = CreateEvent(0, 0, 0, 0);
    IM_ASSERT(event != NULL);

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask                 = 1;

    ID3D12CommandQueue *cmdQueue = NULL;
    hr = g_pd3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue));
    IM_ASSERT(SUCCEEDED(hr));

    ID3D12CommandAllocator *cmdAlloc = NULL;
    hr = g_pd3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              IID_PPV_ARGS(&cmdAlloc));
    IM_ASSERT(SUCCEEDED(hr));

    ID3D12GraphicsCommandList *cmdList = NULL;
    hr = g_pd3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc, NULL,
                                         IID_PPV_ARGS(&cmdList));
    IM_ASSERT(SUCCEEDED(hr));

    cmdList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, NULL);
    cmdList->ResourceBarrier(1, &barrier);

    hr = cmdList->Close();
    IM_ASSERT(SUCCEEDED(hr));

    cmdQueue->ExecuteCommandLists(1, (ID3D12CommandList *const *)&cmdList);
    hr = cmdQueue->Signal(fence, 1);
    IM_ASSERT(SUCCEEDED(hr));

    fence->SetEventOnCompletion(1, event);
    WaitForSingleObject(event, INFINITE);

    cmdList->Release();
    cmdAlloc->Release();
    cmdQueue->Release();
    CloseHandle(event);
    fence->Release();
    uploadBuffer->Release();

    // Create texture view
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format                    = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels       = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Shader4ComponentMapping   = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    g_pd3dDevice->CreateShaderResourceView(pTexture, &srvDesc, g_hFontSrvCpuDescHandle);
    SafeRelease(g_pFontTextureResource);
    g_pFontTextureResource = pTexture;
  }

  // Store our identifier
  static_assert(sizeof(ImTextureID) >= sizeof(g_hFontSrvGpuDescHandle.ptr),
                "Can't pack descriptor handle into TexID, 32-bit not supported yet.");
  io.Fonts->TexID = (ImTextureID)g_hFontSrvGpuDescHandle.ptr;
}

bool ImGui_ImplDX12_CreateDeviceObjects() {
  if (!g_pd3dDevice) return false;
  if (g_pPipelineState) ImGui_ImplDX12_InvalidateDeviceObjects();

  // Create the root signature
  {
    D3D12_DESCRIPTOR_RANGE descRange            = {};
    descRange.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    descRange.NumDescriptors                    = 1;
    descRange.BaseShaderRegister                = 0;
    descRange.RegisterSpace                     = 0;
    descRange.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER param[2] = {};

    param[0].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param[0].Constants.ShaderRegister = 0;
    param[0].Constants.RegisterSpace  = 0;
    param[0].Constants.Num32BitValues = 16;
    param[0].ShaderVisibility         = D3D12_SHADER_VISIBILITY_VERTEX;

    param[1].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param[1].DescriptorTable.NumDescriptorRanges = 1;
    param[1].DescriptorTable.pDescriptorRanges   = &descRange;
    param[1].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC staticSampler = {};
    staticSampler.Filter                    = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSampler.AddressU                  = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSampler.AddressV                  = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSampler.AddressW                  = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSampler.MipLODBias                = 0.f;
    staticSampler.MaxAnisotropy             = 0;
    staticSampler.ComparisonFunc            = D3D12_COMPARISON_FUNC_ALWAYS;
    staticSampler.BorderColor               = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    staticSampler.MinLOD                    = 0.f;
    staticSampler.MaxLOD                    = 0.f;
    staticSampler.ShaderRegister            = 0;
    staticSampler.RegisterSpace             = 0;
    staticSampler.ShaderVisibility          = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters             = _countof(param);
    desc.pParameters               = param;
    desc.NumStaticSamplers         = 1;
    desc.pStaticSamplers           = &staticSampler;
    desc.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                 D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                 D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                 D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

    ID3DBlob *blob = NULL;
    if (D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL) != S_OK)
      return false;

    g_pd3dDevice->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                      IID_PPV_ARGS(&g_pRootSignature));
    blob->Release();
  }

  // By using D3DCompile() from <d3dcompiler.h> / d3dcompiler.lib, we introduce a dependency to a
  // given version of d3dcompiler_XX.dll (see D3DCOMPILER_DLL_A) If you would like to use this DX12
  // sample code but remove this dependency you can:
  //  1) compile once, save the compiled shader blobs into a file or source code and pass them to
  //  CreateVertexShader()/CreatePixelShader() [preferred solution] 2) use code to detect any
  //  version of the DLL and grab a pointer to D3DCompile from the DLL.
  // See https://github.com/ocornut/imgui/pull/638 for sources and details.

  D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
  memset(&psoDesc, 0, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
  psoDesc.NodeMask              = 1;
  psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  psoDesc.pRootSignature        = g_pRootSignature;
  psoDesc.SampleMask            = UINT_MAX;
  psoDesc.NumRenderTargets      = 1;
  psoDesc.RTVFormats[0]         = g_RTVFormat;
  psoDesc.SampleDesc.Count      = 1;
  psoDesc.Flags                 = D3D12_PIPELINE_STATE_FLAG_NONE;

  ID3DBlob *vertexShaderBlob;
  ID3DBlob *pixelShaderBlob;

  // Create the vertex shader
  {
    static const char *vertexShader = "cbuffer vertexBuffer : register(b0) \
            {\
              float4x4 ProjectionMatrix; \
            };\
            struct VS_INPUT\
            {\
              float2 pos : POSITION;\
              float4 col : COLOR0;\
              float2 uv  : TEXCOORD0;\
            };\
            \
            struct PS_INPUT\
            {\
              float4 pos : SV_POSITION;\
              float4 col : COLOR0;\
              float2 uv  : TEXCOORD0;\
            };\
            \
            PS_INPUT main(VS_INPUT input)\
            {\
              PS_INPUT output;\
              output.pos = mul( ProjectionMatrix, float4(input.pos.xy, 0.f, 1.f));\
              output.col = input.col;\
              output.uv  = input.uv;\
              return output;\
            }";

    if (FAILED(D3DCompile(vertexShader, strlen(vertexShader), NULL, NULL, NULL, "main", "vs_5_0", 0,
                          0, &vertexShaderBlob, NULL)))
      return false; // NB: Pass ID3D10Blob* pErrorBlob to D3DCompile() to get error showing in
                    // (const char*)pErrorBlob->GetBufferPointer(). Make sure to Release() the blob!
    psoDesc.VS = {vertexShaderBlob->GetBufferPointer(), vertexShaderBlob->GetBufferSize()};

    // Create the input layout
    static D3D12_INPUT_ELEMENT_DESC local_layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, (UINT)IM_OFFSETOF(ImDrawVert, pos),
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, (UINT)IM_OFFSETOF(ImDrawVert, uv),
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, (UINT)IM_OFFSETOF(ImDrawVert, col),
         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    psoDesc.InputLayout = {local_layout, 3};
  }

  // Create the pixel shader
  {
    static const char *pixelShader = "struct PS_INPUT\
            {\
              float4 pos : SV_POSITION;\
              float4 col : COLOR0;\
              float2 uv  : TEXCOORD0;\
            };\
            SamplerState sampler0 : register(s0);\
            Texture2D texture0 : register(t0);\
            \
            float4 main(PS_INPUT input) : SV_Target\
            {\
              float4 out_col = input.col * texture0.Sample(sampler0, input.uv); \
              return out_col; \
            }";

    if (FAILED(D3DCompile(pixelShader, strlen(pixelShader), NULL, NULL, NULL, "main", "ps_5_0", 0,
                          0, &pixelShaderBlob, NULL))) {
      vertexShaderBlob->Release();
      return false; // NB: Pass ID3D10Blob* pErrorBlob to D3DCompile() to get error showing in
                    // (const char*)pErrorBlob->GetBufferPointer(). Make sure to Release() the blob!
    }
    psoDesc.PS = {pixelShaderBlob->GetBufferPointer(), pixelShaderBlob->GetBufferSize()};
  }

  // Create the blending setup
  {
    D3D12_BLEND_DESC &desc                     = psoDesc.BlendState;
    desc.AlphaToCoverageEnable                 = false;
    desc.RenderTarget[0].BlendEnable           = true;
    desc.RenderTarget[0].SrcBlend              = D3D12_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].DestBlend             = D3D12_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].BlendOp               = D3D12_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlendAlpha         = D3D12_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha        = D3D12_BLEND_ZERO;
    desc.RenderTarget[0].BlendOpAlpha          = D3D12_BLEND_OP_ADD;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
  }

  // Create the rasterizer state
  {
    D3D12_RASTERIZER_DESC &desc = psoDesc.RasterizerState;
    desc.FillMode               = D3D12_FILL_MODE_SOLID;
    desc.CullMode               = D3D12_CULL_MODE_NONE;
    desc.FrontCounterClockwise  = FALSE;
    desc.DepthBias              = D3D12_DEFAULT_DEPTH_BIAS;
    desc.DepthBiasClamp         = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    desc.SlopeScaledDepthBias   = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    desc.DepthClipEnable        = true;
    desc.MultisampleEnable      = FALSE;
    desc.AntialiasedLineEnable  = FALSE;
    desc.ForcedSampleCount      = 0;
    desc.ConservativeRaster     = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
  }

  // Create depth-stencil State
  {
    D3D12_DEPTH_STENCIL_DESC &desc   = psoDesc.DepthStencilState;
    desc.DepthEnable                 = false;
    desc.DepthWriteMask              = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthFunc                   = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.StencilEnable               = false;
    desc.FrontFace.StencilFailOp     = desc.FrontFace.StencilDepthFailOp =
        desc.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    desc.FrontFace.StencilFunc       = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.BackFace                    = desc.FrontFace;
  }

  HRESULT result_pipeline_state =
      g_pd3dDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&g_pPipelineState));
  vertexShaderBlob->Release();
  pixelShaderBlob->Release();
  if (result_pipeline_state != S_OK) return false;

  ImGui_ImplDX12_CreateFontsTexture();

  return true;
}

void ImGui_ImplDX12_InvalidateDeviceObjects() {
  if (!ImGui::GetCurrentContext()) return;
  if (!g_pd3dDevice) return;

  SafeRelease(g_pRootSignature);
  SafeRelease(g_pPipelineState);
  SafeRelease(g_pFontTextureResource);

  ImGuiIO &io = ImGui::GetIO();
  io.Fonts->TexID =
      NULL; // We copied g_pFontTextureView to io.Fonts->TexID so let's clear that as well.
}

bool ImGui_ImplDX12_Init(ID3D12Device *device, int num_frames_in_flight, DXGI_FORMAT rtv_format,
                         ID3D12DescriptorHeap *      cbv_srv_heap,
                         D3D12_CPU_DESCRIPTOR_HANDLE font_srv_cpu_desc_handle,
                         D3D12_GPU_DESCRIPTOR_HANDLE font_srv_gpu_desc_handle) {
  // Setup back-end capabilities flags
  ImGuiIO &io            = ImGui::GetIO();
  io.BackendRendererName = "imgui_impl_dx12";
  io.BackendFlags |=
      ImGuiBackendFlags_RendererHasVtxOffset; // We can honor the ImDrawCmd::VtxOffset field,
                                              // allowing for large meshes.
  io.BackendFlags |=
      ImGuiBackendFlags_RendererHasViewports; // We can create multi-viewports on the Renderer side
                                              // (optional) // FIXME-VIEWPORT: Actually unfinished..

  g_pd3dDevice            = device;
  g_RTVFormat             = rtv_format;
  g_hFontSrvCpuDescHandle = font_srv_cpu_desc_handle;
  g_hFontSrvGpuDescHandle = font_srv_gpu_desc_handle;
  g_numFramesInFlight     = num_frames_in_flight;
  g_pd3dSrvDescHeap       = cbv_srv_heap;

  // Create a dummy ImGuiViewportDataDx12 holder for the main viewport,
  // Since this is created and managed by the application, we will only use the ->Resources[]
  // fields.
  ImGuiViewport *main_viewport    = ImGui::GetMainViewport();
  main_viewport->RendererUserData = IM_NEW(ImGuiViewportDataDx12)();

  // Setup back-end capabilities flags
  io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports; // We can create multi-viewports on the
                                                             // Renderer side (optional)
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) ImGui_ImplDX12_InitPlatformInterface();

  return true;
}

void ImGui_ImplDX12_Shutdown() {
  // Manually delete main viewport render resources in-case we haven't initialized for viewports
  ImGuiViewport *main_viewport = ImGui::GetMainViewport();
  if (ImGuiViewportDataDx12 *data = (ImGuiViewportDataDx12 *)main_viewport->RendererUserData) {
    // We could just call ImGui_ImplDX12_DestroyWindow(main_viewport) as a convenience but that
    // would be misleading since we only use data->Resources[]
    for (UINT i = 0; i < g_numFramesInFlight; i++)
      ImGui_ImplDX12_DestroyRenderBuffers(&data->FrameRenderBuffers[i]);
    IM_DELETE(data);
    main_viewport->RendererUserData = NULL;
  }

  // Clean up windows and device objects
  ImGui_ImplDX12_ShutdownPlatformInterface();
  ImGui_ImplDX12_InvalidateDeviceObjects();

  g_pd3dDevice                = NULL;
  g_hFontSrvCpuDescHandle.ptr = 0;
  g_hFontSrvGpuDescHandle.ptr = 0;
  g_numFramesInFlight         = 0;
  g_pd3dSrvDescHeap           = NULL;
}

void ImGui_ImplDX12_NewFrame() {
  if (!g_pPipelineState) ImGui_ImplDX12_CreateDeviceObjects();
}

//--------------------------------------------------------------------------------------------------------
// MULTI-VIEWPORT / PLATFORM INTERFACE SUPPORT
// This is an _advanced_ and _optional_ feature, allowing the back-end to create and handle multiple
// viewports simultaneously. If you are new to dear imgui or creating a new binding for dear imgui,
// it is recommended that you completely ignore this section first..
//--------------------------------------------------------------------------------------------------------

static void ImGui_ImplDX12_CreateWindow(ImGuiViewport *viewport) {
  ImGuiViewportDataDx12 *data = IM_NEW(ImGuiViewportDataDx12)();
  viewport->RendererUserData  = data;

  // PlatformHandleRaw should always be a HWND, whereas PlatformHandle might be a higher-level
  // handle (e.g. GLFWWindow*, SDL_Window*). Some back-ends will leave PlatformHandleRaw NULL, in
  // which case we assume PlatformHandle will contain the HWND.
  HWND hwnd = viewport->PlatformHandleRaw ? (HWND)viewport->PlatformHandleRaw
                                          : (HWND)viewport->PlatformHandle;
  IM_ASSERT(hwnd != 0);

  data->FrameIndex = UINT_MAX;

  // Create command queue.
  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
  queue_desc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;

  HRESULT res = S_OK;
  res         = g_pd3dDevice->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&data->CommandQueue));
  IM_ASSERT(res == S_OK);

  // Create command allocator.
  for (UINT i = 0; i < g_numFramesInFlight; ++i) {
    res = g_pd3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               IID_PPV_ARGS(&data->FrameCtx[i].CommandAllocator));
    IM_ASSERT(res == S_OK);
  }

  // Create command list.
  res = g_pd3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        data->FrameCtx[0].CommandAllocator, NULL,
                                        IID_PPV_ARGS(&data->CommandList));
  IM_ASSERT(res == S_OK);
  data->CommandList->Close();

  // Create fence.
  res = g_pd3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&data->Fence));
  IM_ASSERT(res == S_OK);

  data->FenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  IM_ASSERT(data->FenceEvent != NULL);

  // Create swap chain
  // FIXME-VIEWPORT: May want to copy/inherit swap chain settings from the user/application.
  DXGI_SWAP_CHAIN_DESC1 sd1;
  ZeroMemory(&sd1, sizeof(sd1));
  sd1.BufferCount        = g_numFramesInFlight;
  sd1.Width              = (UINT)viewport->Size.x;
  sd1.Height             = (UINT)viewport->Size.y;
  sd1.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd1.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd1.SampleDesc.Count   = 1;
  sd1.SampleDesc.Quality = 0;
  sd1.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  sd1.AlphaMode          = DXGI_ALPHA_MODE_UNSPECIFIED;
  sd1.Scaling            = DXGI_SCALING_STRETCH;
  sd1.Stereo             = FALSE;

  IDXGIFactory4 *dxgi_factory = NULL;
  res                         = ::CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
  IM_ASSERT(res == S_OK);

  IDXGISwapChain1 *swap_chain = NULL;
  res =
      dxgi_factory->CreateSwapChainForHwnd(data->CommandQueue, hwnd, &sd1, NULL, NULL, &swap_chain);
  IM_ASSERT(res == S_OK);

  dxgi_factory->Release();

  // Or swapChain.As(&mSwapChain)
  IM_ASSERT(data->SwapChain == NULL);
  swap_chain->QueryInterface(IID_PPV_ARGS(&data->SwapChain));
  swap_chain->Release();

  // Create the render targets
  if (data->SwapChain) {
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    desc.NumDescriptors             = g_numFramesInFlight;
    desc.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    desc.NodeMask                   = 1;

    HRESULT hr = g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&data->RtvDescHeap));
    IM_ASSERT(hr == S_OK);

    SIZE_T rtv_descriptor_size =
        g_pd3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle =
        data->RtvDescHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < g_numFramesInFlight; i++) {
      data->FrameCtx[i].RenderTargetCpuDescriptors = rtv_handle;
      rtv_handle.ptr += rtv_descriptor_size;
    }

    ID3D12Resource *back_buffer;
    for (UINT i = 0; i < g_numFramesInFlight; i++) {
      IM_ASSERT(data->FrameCtx[i].RenderTarget == NULL);
      data->SwapChain->GetBuffer(i, IID_PPV_ARGS(&back_buffer));
      g_pd3dDevice->CreateRenderTargetView(back_buffer, NULL,
                                           data->FrameCtx[i].RenderTargetCpuDescriptors);
      data->FrameCtx[i].RenderTarget = back_buffer;
    }
  }

  for (UINT i = 0; i < g_numFramesInFlight; i++)
    ImGui_ImplDX12_DestroyRenderBuffers(&data->FrameRenderBuffers[i]);
}

static void ImGui_WaitForPendingOperations(ImGuiViewportDataDx12 *data) {
  HRESULT hr = S_FALSE;
  if (data && data->CommandQueue && data->Fence && data->FenceEvent) {
    hr = data->CommandQueue->Signal(data->Fence, ++data->FenceSignaledValue);
    IM_ASSERT(hr == S_OK);
    ::WaitForSingleObject(data->FenceEvent, 0); // Reset any forgotten waits
    hr = data->Fence->SetEventOnCompletion(data->FenceSignaledValue, data->FenceEvent);
    IM_ASSERT(hr == S_OK);
    ::WaitForSingleObject(data->FenceEvent, INFINITE);
  }
}

static void ImGui_ImplDX12_DestroyWindow(ImGuiViewport *viewport) {
  // The main viewport (owned by the application) will always have RendererUserData == NULL since we
  // didn't create the data for it.
  if (ImGuiViewportDataDx12 *data = (ImGuiViewportDataDx12 *)viewport->RendererUserData) {
    ImGui_WaitForPendingOperations(data);

    SafeRelease(data->CommandQueue);
    SafeRelease(data->CommandList);
    SafeRelease(data->SwapChain);
    SafeRelease(data->RtvDescHeap);
    SafeRelease(data->Fence);
    ::CloseHandle(data->FenceEvent);
    data->FenceEvent = NULL;

    for (UINT i = 0; i < g_numFramesInFlight; i++) {
      SafeRelease(data->FrameCtx[i].RenderTarget);
      SafeRelease(data->FrameCtx[i].CommandAllocator);
      ImGui_ImplDX12_DestroyRenderBuffers(&data->FrameRenderBuffers[i]);
    }
    IM_DELETE(data);
  }
  viewport->RendererUserData = NULL;
}

static void ImGui_ImplDX12_SetWindowSize(ImGuiViewport *viewport, ImVec2 size) {
  ImGuiViewportDataDx12 *data = (ImGuiViewportDataDx12 *)viewport->RendererUserData;

  ImGui_WaitForPendingOperations(data);

  for (UINT i = 0; i < g_numFramesInFlight; i++) SafeRelease(data->FrameCtx[i].RenderTarget);

  if (data->SwapChain) {
    ID3D12Resource *back_buffer = NULL;
    data->SwapChain->ResizeBuffers(0, (UINT)size.x, (UINT)size.y, DXGI_FORMAT_UNKNOWN, 0);
    for (UINT i = 0; i < g_numFramesInFlight; i++) {
      data->SwapChain->GetBuffer(i, IID_PPV_ARGS(&back_buffer));
      g_pd3dDevice->CreateRenderTargetView(back_buffer, NULL,
                                           data->FrameCtx[i].RenderTargetCpuDescriptors);
      data->FrameCtx[i].RenderTarget = back_buffer;
    }
  }
}

static void ImGui_ImplDX12_RenderWindow(ImGuiViewport *viewport, void *) {
  ImGuiViewportDataDx12 *data = (ImGuiViewportDataDx12 *)viewport->RendererUserData;

  ImGui_ImplDX12_FrameContext *frame_context =
      &data->FrameCtx[data->FrameIndex % g_numFramesInFlight];
  UINT back_buffer_idx = data->SwapChain->GetCurrentBackBufferIndex();

  const ImVec4           clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
  D3D12_RESOURCE_BARRIER barrier     = {};
  barrier.Type                       = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags                      = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource       = data->FrameCtx[back_buffer_idx].RenderTarget;
  barrier.Transition.Subresource     = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore     = D3D12_RESOURCE_STATE_PRESENT;
  barrier.Transition.StateAfter      = D3D12_RESOURCE_STATE_RENDER_TARGET;

  // Draw
  ID3D12GraphicsCommandList *cmd_list = data->CommandList;

  frame_context->CommandAllocator->Reset();
  cmd_list->Reset(frame_context->CommandAllocator, NULL);
  cmd_list->ResourceBarrier(1, &barrier);
  cmd_list->OMSetRenderTargets(1, &data->FrameCtx[back_buffer_idx].RenderTargetCpuDescriptors,
                               FALSE, NULL);
  if (!(viewport->Flags & ImGuiViewportFlags_NoRendererClear))
    cmd_list->ClearRenderTargetView(data->FrameCtx[back_buffer_idx].RenderTargetCpuDescriptors,
                                    (float *)&clear_color, 0, NULL);
  cmd_list->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);

  ImGui_ImplDX12_RenderDrawData(viewport->DrawData, cmd_list);

  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
  barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
  cmd_list->ResourceBarrier(1, &barrier);
  cmd_list->Close();

  data->CommandQueue->Wait(data->Fence, data->FenceSignaledValue);
  data->CommandQueue->ExecuteCommandLists(1, (ID3D12CommandList *const *)&cmd_list);
  data->CommandQueue->Signal(data->Fence, ++data->FenceSignaledValue);
}

static void ImGui_ImplDX12_SwapBuffers(ImGuiViewport *viewport, void *) {
  ImGuiViewportDataDx12 *data = (ImGuiViewportDataDx12 *)viewport->RendererUserData;

  data->SwapChain->Present(0, 0);
  while (data->Fence->GetCompletedValue() < data->FenceSignaledValue) ::SwitchToThread();
}

void ImGui_ImplDX12_InitPlatformInterface() {
  ImGuiPlatformIO &platform_io       = ImGui::GetPlatformIO();
  platform_io.Renderer_CreateWindow  = ImGui_ImplDX12_CreateWindow;
  platform_io.Renderer_DestroyWindow = ImGui_ImplDX12_DestroyWindow;
  platform_io.Renderer_SetWindowSize = ImGui_ImplDX12_SetWindowSize;
  platform_io.Renderer_RenderWindow  = ImGui_ImplDX12_RenderWindow;
  platform_io.Renderer_SwapBuffers   = ImGui_ImplDX12_SwapBuffers;
}

void ImGui_ImplDX12_ShutdownPlatformInterface() { ImGui::DestroyPlatformWindows(); }

// dear imgui: Platform Binding for Windows (standard windows API for 32 and 64 bits applications)
// This needs to be used along with a Renderer (e.g. DirectX11, OpenGL3, Vulkan..)

// Implemented features:
//  [X] Platform: Clipboard support (for Win32 this is actually part of core dear imgui)
//  [X] Platform: Mouse cursor shape and visibility. Disable with 'io.ConfigFlags |=
//  ImGuiConfigFlags_NoMouseCursorChange'. [X] Platform: Keyboard arrays indexed using VK_* Virtual
//  Key Codes, e.g. ImGui::IsKeyPressed(VK_SPACE). [X] Platform: Gamepad support. Enabled with
//  'io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad'. [X] Platform: Multi-viewport support
//  (multiple windows). Enable with 'io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable'.

#  include "imgui.h"
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <tchar.h>
#  include <windows.h>

// Using XInput library for gamepad (with recent Windows SDK this may leads to executables which
// won't run on Windows 7)
#  ifndef IMGUI_IMPL_WIN32_DISABLE_GAMEPAD
#    include <XInput.h>
#  else
#    define IMGUI_IMPL_WIN32_DISABLE_LINKING_XINPUT
#  endif
#  if defined(_MSC_VER) && !defined(IMGUI_IMPL_WIN32_DISABLE_LINKING_XINPUT)
#    pragma comment(lib, "xinput")
//#pragma comment(lib, "Xinput9_1_0")
#  endif

// CHANGELOG
// (minor and older changes stripped away, please see git history for details)
//  2020-XX-XX: Platform: Added support for multiple windows via the ImGuiPlatformIO interface.
//  2020-03-03: Inputs: Calling AddInputCharacterUTF16() to support surrogate pairs leading to
//  codepoint >= 0x10000 (for more complete CJK inputs) 2020-02-17: Added
//  ImGui_ImplWin32_EnableDpiAwareness(), ImGui_ImplWin32_GetDpiScaleForHwnd(),
//  ImGui_ImplWin32_GetDpiScaleForMonitor() helper functions. 2020-01-14: Inputs: Added support for
//  #define IMGUI_IMPL_WIN32_DISABLE_GAMEPAD/IMGUI_IMPL_WIN32_DISABLE_LINKING_XINPUT. 2019-12-05:
//  Inputs: Added support for ImGuiMouseCursor_NotAllowed mouse cursor. 2019-05-11: Inputs: Don't
//  filter value from WM_CHAR before calling AddInputCharacter(). 2019-01-17: Misc: Using
//  GetForegroundWindow()+IsChild() instead of GetActiveWindow() to be compatible with windows
//  created in a different thread or parent. 2019-01-17: Inputs: Added support for mouse buttons 4
//  and 5 via WM_XBUTTON* messages. 2019-01-15: Inputs: Added support for XInput gamepads (if
//  ImGuiConfigFlags_NavEnableGamepad is set by user application). 2018-11-30: Misc: Setting up
//  io.BackendPlatformName so it can be displayed in the About Window. 2018-06-29: Inputs: Added
//  support for the ImGuiMouseCursor_Hand cursor. 2018-06-10: Inputs: Fixed handling of mouse wheel
//  messages to support fine position messages (typically sent by track-pads). 2018-06-08: Misc:
//  Extracted imgui_impl_win32.cpp/.h away from the old combined DX9/DX10/DX11/DX12 examples.
//  2018-03-20: Misc: Setup io.BackendFlags ImGuiBackendFlags_HasMouseCursors and
//  ImGuiBackendFlags_HasSetMousePos flags + honor ImGuiConfigFlags_NoMouseCursorChange flag.
//  2018-02-20: Inputs: Added support for mouse cursors (ImGui::GetMouseCursor() value and
//  WM_SETCURSOR message handling). 2018-02-06: Inputs: Added mapping for ImGuiKey_Space.
//  2018-02-06: Inputs: Honoring the io.WantSetMousePos by repositioning the mouse (when using
//  navigation and ImGuiConfigFlags_NavMoveMouse is set). 2018-02-06: Misc: Removed call to
//  ImGui::Shutdown() which is not available from 1.60 WIP, user needs to call
//  CreateContext/DestroyContext themselves. 2018-01-20: Inputs: Added Horizontal Mouse Wheel
//  support. 2018-01-08: Inputs: Added mapping for ImGuiKey_Insert. 2018-01-05: Inputs: Added
//  WM_LBUTTONDBLCLK double-click handlers for window classes with the CS_DBLCLKS flag. 2017-10-23:
//  Inputs: Added WM_SYSKEYDOWN / WM_SYSKEYUP handlers so e.g. the VK_MENU key can be read.
//  2017-10-23: Inputs: Using Win32 ::SetCapture/::GetCapture() to retrieve mouse positions outside
//  the client area when dragging. 2016-11-12: Inputs: Only call Win32 ::SetCursor(NULL) when
//  io.MouseDrawCursor is set.

// Win32 Data
static HWND             g_hWnd                 = NULL;
static INT64            g_Time                 = 0;
static INT64            g_TicksPerSecond       = 0;
static ImGuiMouseCursor g_LastMouseCursor      = ImGuiMouseCursor_COUNT;
static bool             g_HasGamepad           = false;
static bool             g_WantUpdateHasGamepad = true;
static bool             g_WantUpdateMonitors   = true;

// Forward Declarations
static void ImGui_ImplWin32_InitPlatformInterface();
static void ImGui_ImplWin32_ShutdownPlatformInterface();
static void ImGui_ImplWin32_UpdateMonitors();

// Functions
bool ImGui_ImplWin32_Init(void *hwnd) {
  if (!::QueryPerformanceFrequency((LARGE_INTEGER *)&g_TicksPerSecond)) return false;
  if (!::QueryPerformanceCounter((LARGE_INTEGER *)&g_Time)) return false;

  // Setup back-end capabilities flags
  ImGuiIO &io = ImGui::GetIO();
  io.BackendFlags |=
      ImGuiBackendFlags_HasMouseCursors; // We can honor GetMouseCursor() values (optional)
  io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos; // We can honor io.WantSetMousePos requests
                                                       // (optional, rarely used)
  io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports; // We can create multi-viewports on the
                                                             // Platform side (optional)
  io.BackendFlags |=
      ImGuiBackendFlags_HasMouseHoveredViewport; // We can set io.MouseHoveredViewport correctly
                                                 // (optional, not easy)
  io.BackendPlatformName = "imgui_impl_win32";

  // Our mouse update function expect PlatformHandle to be filled for the main viewport
  g_hWnd                        = (HWND)hwnd;
  ImGuiViewport *main_viewport  = ImGui::GetMainViewport();
  main_viewport->PlatformHandle = main_viewport->PlatformHandleRaw = (void *)g_hWnd;
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) ImGui_ImplWin32_InitPlatformInterface();

  // Keyboard mapping. ImGui will use those indices to peek into the io.KeysDown[] array that we
  // will update during the application lifetime.
  io.KeyMap[ImGuiKey_Tab]         = VK_TAB;
  io.KeyMap[ImGuiKey_LeftArrow]   = VK_LEFT;
  io.KeyMap[ImGuiKey_RightArrow]  = VK_RIGHT;
  io.KeyMap[ImGuiKey_UpArrow]     = VK_UP;
  io.KeyMap[ImGuiKey_DownArrow]   = VK_DOWN;
  io.KeyMap[ImGuiKey_PageUp]      = VK_PRIOR;
  io.KeyMap[ImGuiKey_PageDown]    = VK_NEXT;
  io.KeyMap[ImGuiKey_Home]        = VK_HOME;
  io.KeyMap[ImGuiKey_End]         = VK_END;
  io.KeyMap[ImGuiKey_Insert]      = VK_INSERT;
  io.KeyMap[ImGuiKey_Delete]      = VK_DELETE;
  io.KeyMap[ImGuiKey_Backspace]   = VK_BACK;
  io.KeyMap[ImGuiKey_Space]       = VK_SPACE;
  io.KeyMap[ImGuiKey_Enter]       = VK_RETURN;
  io.KeyMap[ImGuiKey_Escape]      = VK_ESCAPE;
  io.KeyMap[ImGuiKey_KeyPadEnter] = VK_RETURN;
  io.KeyMap[ImGuiKey_A]           = 'A';
  io.KeyMap[ImGuiKey_C]           = 'C';
  io.KeyMap[ImGuiKey_V]           = 'V';
  io.KeyMap[ImGuiKey_X]           = 'X';
  io.KeyMap[ImGuiKey_Y]           = 'Y';
  io.KeyMap[ImGuiKey_Z]           = 'Z';

  return true;
}

void ImGui_ImplWin32_Shutdown() {
  ImGui_ImplWin32_ShutdownPlatformInterface();
  g_hWnd = (HWND)0;
}

static bool ImGui_ImplWin32_UpdateMouseCursor() {
  ImGuiIO &io = ImGui::GetIO();
  if (io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange) return false;

  ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();
  if (imgui_cursor == ImGuiMouseCursor_None || io.MouseDrawCursor) {
    // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor
    ::SetCursor(NULL);
  } else {
    // Show OS mouse cursor
    LPTSTR win32_cursor = IDC_ARROW;
    switch (imgui_cursor) {
    case ImGuiMouseCursor_Arrow: win32_cursor = IDC_ARROW; break;
    case ImGuiMouseCursor_TextInput: win32_cursor = IDC_IBEAM; break;
    case ImGuiMouseCursor_ResizeAll: win32_cursor = IDC_SIZEALL; break;
    case ImGuiMouseCursor_ResizeEW: win32_cursor = IDC_SIZEWE; break;
    case ImGuiMouseCursor_ResizeNS: win32_cursor = IDC_SIZENS; break;
    case ImGuiMouseCursor_ResizeNESW: win32_cursor = IDC_SIZENESW; break;
    case ImGuiMouseCursor_ResizeNWSE: win32_cursor = IDC_SIZENWSE; break;
    case ImGuiMouseCursor_Hand: win32_cursor = IDC_HAND; break;
    case ImGuiMouseCursor_NotAllowed: win32_cursor = IDC_NO; break;
    }
    ::SetCursor(::LoadCursor(NULL, win32_cursor));
  }
  return true;
}

// This code supports multi-viewports (multiple OS Windows mapped into different Dear ImGui
// viewports) Because of that, it is a little more complicated than your typical single-viewport
// binding code!
static void ImGui_ImplWin32_UpdateMousePos() {
  ImGuiIO &io = ImGui::GetIO();

  // Set OS mouse position if requested (rarely used, only when
  // ImGuiConfigFlags_NavEnableSetMousePos is enabled by user) (When multi-viewports are enabled,
  // all imgui positions are same as OS positions)
  if (io.WantSetMousePos) {
    POINT pos = {(int)io.MousePos.x, (int)io.MousePos.y};
    if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) == 0) ::ClientToScreen(g_hWnd, &pos);
    ::SetCursorPos(pos.x, pos.y);
  }

  io.MousePos             = ImVec2(-FLT_MAX, -FLT_MAX);
  io.MouseHoveredViewport = 0;

  // Set imgui mouse position
  POINT mouse_screen_pos;
  if (!::GetCursorPos(&mouse_screen_pos)) return;
  if (HWND focused_hwnd = ::GetForegroundWindow()) {
    if (::IsChild(focused_hwnd, g_hWnd)) focused_hwnd = g_hWnd;
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      // Multi-viewport mode: mouse position in OS absolute coordinates (io.MousePos is (0,0) when
      // the mouse is on the upper-left of the primary monitor) This is the position you can get
      // with GetCursorPos(). In theory adding viewport->Pos is also the reverse operation of doing
      // ScreenToClient().
      if (ImGui::FindViewportByPlatformHandle((void *)focused_hwnd) != NULL)
        io.MousePos = ImVec2((float)mouse_screen_pos.x, (float)mouse_screen_pos.y);
    } else {
      // Single viewport mode: mouse position in client window coordinates (io.MousePos is (0,0)
      // when the mouse is on the upper-left corner of the app window.) This is the position you can
      // get with GetCursorPos() + ScreenToClient() or from WM_MOUSEMOVE.
      if (focused_hwnd == g_hWnd) {
        POINT mouse_client_pos = mouse_screen_pos;
        ::ScreenToClient(focused_hwnd, &mouse_client_pos);
        io.MousePos = ImVec2((float)mouse_client_pos.x, (float)mouse_client_pos.y);
      }
    }
  }

  // (Optional) When using multiple viewports: set io.MouseHoveredViewport to the viewport the OS
  // mouse cursor is hovering. Important: this information is not easy to provide and many
  // high-level windowing library won't be able to provide it correctly, because
  // - This is _ignoring_ viewports with the ImGuiViewportFlags_NoInputs flag (pass-through
  // windows).
  // - This is _regardless_ of whether another viewport is focused or being dragged from.
  // If ImGuiBackendFlags_HasMouseHoveredViewport is not set by the back-end, imgui will ignore this
  // field and infer the information by relying on the rectangles and last focused time of every
  // viewports it knows about. It will be unaware of foreign windows that may be sitting between or
  // over your windows.
  if (HWND hovered_hwnd = ::WindowFromPoint(mouse_screen_pos))
    if (ImGuiViewport *viewport = ImGui::FindViewportByPlatformHandle((void *)hovered_hwnd))
      if ((viewport->Flags & ImGuiViewportFlags_NoInputs) ==
          0) // FIXME: We still get our NoInputs window with WM_NCHITTEST/HTTRANSPARENT code when
             // decorated?
        io.MouseHoveredViewport = viewport->ID;
}

// Gamepad navigation mapping
static void ImGui_ImplWin32_UpdateGamepads() {
#  ifndef IMGUI_IMPL_WIN32_DISABLE_GAMEPAD
  ImGuiIO &io = ImGui::GetIO();
  memset(io.NavInputs, 0, sizeof(io.NavInputs));
  if ((io.ConfigFlags & ImGuiConfigFlags_NavEnableGamepad) == 0) return;

  // Calling XInputGetState() every frame on disconnected gamepads is unfortunately too slow.
  // Instead we refresh gamepad availability by calling XInputGetCapabilities() _only_ after
  // receiving WM_DEVICECHANGE.
  if (g_WantUpdateHasGamepad) {
    XINPUT_CAPABILITIES caps;
    g_HasGamepad = (XInputGetCapabilities(0, XINPUT_FLAG_GAMEPAD, &caps) == ERROR_SUCCESS);
    g_WantUpdateHasGamepad = false;
  }

  XINPUT_STATE xinput_state;
  io.BackendFlags &= ~ImGuiBackendFlags_HasGamepad;
  if (g_HasGamepad && XInputGetState(0, &xinput_state) == ERROR_SUCCESS) {
    const XINPUT_GAMEPAD &gamepad = xinput_state.Gamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;

#    define MAP_BUTTON(NAV_NO, BUTTON_ENUM)                                                        \
      { io.NavInputs[NAV_NO] = (gamepad.wButtons & BUTTON_ENUM) ? 1.0f : 0.0f; }
#    define MAP_ANALOG(NAV_NO, VALUE, V0, V1)                                                      \
      {                                                                                            \
        float vn = (float)(VALUE - V0) / (float)(V1 - V0);                                         \
        if (vn > 1.0f) vn = 1.0f;                                                                  \
        if (vn > 0.0f && io.NavInputs[NAV_NO] < vn) io.NavInputs[NAV_NO] = vn;                     \
      }
    MAP_BUTTON(ImGuiNavInput_Activate, XINPUT_GAMEPAD_A);               // Cross / A
    MAP_BUTTON(ImGuiNavInput_Cancel, XINPUT_GAMEPAD_B);                 // Circle / B
    MAP_BUTTON(ImGuiNavInput_Menu, XINPUT_GAMEPAD_X);                   // Square / X
    MAP_BUTTON(ImGuiNavInput_Input, XINPUT_GAMEPAD_Y);                  // Triangle / Y
    MAP_BUTTON(ImGuiNavInput_DpadLeft, XINPUT_GAMEPAD_DPAD_LEFT);       // D-Pad Left
    MAP_BUTTON(ImGuiNavInput_DpadRight, XINPUT_GAMEPAD_DPAD_RIGHT);     // D-Pad Right
    MAP_BUTTON(ImGuiNavInput_DpadUp, XINPUT_GAMEPAD_DPAD_UP);           // D-Pad Up
    MAP_BUTTON(ImGuiNavInput_DpadDown, XINPUT_GAMEPAD_DPAD_DOWN);       // D-Pad Down
    MAP_BUTTON(ImGuiNavInput_FocusPrev, XINPUT_GAMEPAD_LEFT_SHOULDER);  // L1 / LB
    MAP_BUTTON(ImGuiNavInput_FocusNext, XINPUT_GAMEPAD_RIGHT_SHOULDER); // R1 / RB
    MAP_BUTTON(ImGuiNavInput_TweakSlow, XINPUT_GAMEPAD_LEFT_SHOULDER);  // L1 / LB
    MAP_BUTTON(ImGuiNavInput_TweakFast, XINPUT_GAMEPAD_RIGHT_SHOULDER); // R1 / RB
    MAP_ANALOG(ImGuiNavInput_LStickLeft, gamepad.sThumbLX, -XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
               -32768);
    MAP_ANALOG(ImGuiNavInput_LStickRight, gamepad.sThumbLX, +XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
               +32767);
    MAP_ANALOG(ImGuiNavInput_LStickUp, gamepad.sThumbLY, +XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
               +32767);
    MAP_ANALOG(ImGuiNavInput_LStickDown, gamepad.sThumbLY, -XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
               -32767);
#    undef MAP_BUTTON
#    undef MAP_ANALOG
  }
#  endif // #ifndef IMGUI_IMPL_WIN32_DISABLE_GAMEPAD
}

static BOOL CALLBACK ImGui_ImplWin32_UpdateMonitors_EnumFunc(HMONITOR monitor, HDC, LPRECT,
                                                             LPARAM) {
  MONITORINFO info = {0};
  info.cbSize      = sizeof(MONITORINFO);
  if (!::GetMonitorInfo(monitor, &info)) return TRUE;
  ImGuiPlatformMonitor imgui_monitor;
  imgui_monitor.MainPos  = ImVec2((float)info.rcMonitor.left, (float)info.rcMonitor.top);
  imgui_monitor.MainSize = ImVec2((float)(info.rcMonitor.right - info.rcMonitor.left),
                                  (float)(info.rcMonitor.bottom - info.rcMonitor.top));
  imgui_monitor.WorkPos  = ImVec2((float)info.rcWork.left, (float)info.rcWork.top);
  imgui_monitor.WorkSize = ImVec2((float)(info.rcWork.right - info.rcWork.left),
                                  (float)(info.rcWork.bottom - info.rcWork.top));
  imgui_monitor.DpiScale = ImGui_ImplWin32_GetDpiScaleForMonitor(monitor);
  ImGuiPlatformIO &io    = ImGui::GetPlatformIO();
  if (info.dwFlags & MONITORINFOF_PRIMARY)
    io.Monitors.push_front(imgui_monitor);
  else
    io.Monitors.push_back(imgui_monitor);
  return TRUE;
}

static void ImGui_ImplWin32_UpdateMonitors() {
  ImGui::GetPlatformIO().Monitors.resize(0);
  ::EnumDisplayMonitors(NULL, NULL, ImGui_ImplWin32_UpdateMonitors_EnumFunc, NULL);
  g_WantUpdateMonitors = false;
}

void ImGui_ImplWin32_NewFrame() {
  ImGuiIO &io = ImGui::GetIO();
  IM_ASSERT(io.Fonts->IsBuilt() &&
            "Font atlas not built! It is generally built by the renderer back-end. Missing call to "
            "renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().");

  // Setup display size (every frame to accommodate for window resizing)
  RECT rect;
  ::GetClientRect(g_hWnd, &rect);
  io.DisplaySize = ImVec2((float)(rect.right - rect.left), (float)(rect.bottom - rect.top));
  if (g_WantUpdateMonitors) ImGui_ImplWin32_UpdateMonitors();

  // Setup time step
  INT64 current_time;
  ::QueryPerformanceCounter((LARGE_INTEGER *)&current_time);
  io.DeltaTime = (float)(current_time - g_Time) / g_TicksPerSecond;
  g_Time       = current_time;

  // Read keyboard modifiers inputs
  io.KeyCtrl  = (::GetKeyState(VK_CONTROL) & 0x8000) != 0;
  io.KeyShift = (::GetKeyState(VK_SHIFT) & 0x8000) != 0;
  io.KeyAlt   = (::GetKeyState(VK_MENU) & 0x8000) != 0;
  io.KeySuper = false;
  // io.KeysDown[], io.MousePos, io.MouseDown[], io.MouseWheel: filled by the WndProc handler below.

  // Update OS mouse position
  ImGui_ImplWin32_UpdateMousePos();

  // Update OS mouse cursor with the cursor requested by imgui
  ImGuiMouseCursor mouse_cursor =
      io.MouseDrawCursor ? ImGuiMouseCursor_None : ImGui::GetMouseCursor();
  if (g_LastMouseCursor != mouse_cursor) {
    g_LastMouseCursor = mouse_cursor;
    ImGui_ImplWin32_UpdateMouseCursor();
  }

  // Update game controllers (if enabled and available)
  ImGui_ImplWin32_UpdateGamepads();
}

// Allow compilation with old Windows SDK. MinGW doesn't have default _WIN32_WINNT/WINVER versions.
#  ifndef WM_MOUSEHWHEEL
#    define WM_MOUSEHWHEEL 0x020E
#  endif
#  ifndef DBT_DEVNODES_CHANGED
#    define DBT_DEVNODES_CHANGED 0x0007
#  endif

// Win32 message handler (process Win32 mouse/keyboard inputs, etc.)
// Call from your application's message handler.
// When implementing your own back-end, you can read the io.WantCaptureMouse, io.WantCaptureKeyboard
// flags to tell if Dear ImGui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main
// application. Generally you may always pass all inputs to Dear ImGui, and hide them from your
// application based on those two flags. PS: In this Win32 handler, we use the capture API
// (GetCapture/SetCapture/ReleaseCapture) to be able to read mouse coordinates when dragging mouse
// outside of our window bounds. PS: We treat DBLCLK messages as regular mouse down messages, so
// this code will work on windows classes that have the CS_DBLCLKS flag set. Our own example app
// code doesn't set this flag.
#  if 0
// Copy this line into your .cpp file to forward declare the function.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
#  endif
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam,
                                                      LPARAM lParam) {
  if (ImGui::GetCurrentContext() == NULL) return 0;

  ImGuiIO &io = ImGui::GetIO();
  switch (msg) {
  case WM_LBUTTONDOWN:
  case WM_LBUTTONDBLCLK:
  case WM_RBUTTONDOWN:
  case WM_RBUTTONDBLCLK:
  case WM_MBUTTONDOWN:
  case WM_MBUTTONDBLCLK:
  case WM_XBUTTONDOWN:
  case WM_XBUTTONDBLCLK: {
    int button = 0;
    if (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONDBLCLK) {
      button = 0;
    }
    if (msg == WM_RBUTTONDOWN || msg == WM_RBUTTONDBLCLK) {
      button = 1;
    }
    if (msg == WM_MBUTTONDOWN || msg == WM_MBUTTONDBLCLK) {
      button = 2;
    }
    if (msg == WM_XBUTTONDOWN || msg == WM_XBUTTONDBLCLK) {
      button = (GET_XBUTTON_WPARAM(wParam) == XBUTTON1) ? 3 : 4;
    }
    if (!ImGui::IsAnyMouseDown() && ::GetCapture() == NULL) ::SetCapture(hwnd);
    io.MouseDown[button] = true;
    return 0;
  }
  case WM_LBUTTONUP:
  case WM_RBUTTONUP:
  case WM_MBUTTONUP:
  case WM_XBUTTONUP: {
    int button = 0;
    if (msg == WM_LBUTTONUP) {
      button = 0;
    }
    if (msg == WM_RBUTTONUP) {
      button = 1;
    }
    if (msg == WM_MBUTTONUP) {
      button = 2;
    }
    if (msg == WM_XBUTTONUP) {
      button = (GET_XBUTTON_WPARAM(wParam) == XBUTTON1) ? 3 : 4;
    }
    io.MouseDown[button] = false;
    if (!ImGui::IsAnyMouseDown() && ::GetCapture() == hwnd) ::ReleaseCapture();
    return 0;
  }
  case WM_MOUSEWHEEL:
    io.MouseWheel += (float)GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA;
    return 0;
  case WM_MOUSEHWHEEL:
    io.MouseWheelH += (float)GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA;
    return 0;
  case WM_KEYDOWN:
  case WM_SYSKEYDOWN:
    if (wParam < 256) io.KeysDown[wParam] = 1;
    return 0;
  case WM_KEYUP:
  case WM_SYSKEYUP:
    if (wParam < 256) io.KeysDown[wParam] = 0;
    return 0;
  case WM_CHAR:
    // You can also use ToAscii()+GetKeyboardState() to retrieve characters.
    if (wParam > 0 && wParam < 0x10000) io.AddInputCharacterUTF16((unsigned short)wParam);
    return 0;
  case WM_SETCURSOR:
    if (LOWORD(lParam) == HTCLIENT && ImGui_ImplWin32_UpdateMouseCursor()) return 1;
    return 0;
  case WM_DEVICECHANGE:
    if ((UINT)wParam == DBT_DEVNODES_CHANGED) g_WantUpdateHasGamepad = true;
    return 0;
  case WM_DISPLAYCHANGE: g_WantUpdateMonitors = true; return 0;
  }
  return 0;
}

//--------------------------------------------------------------------------------------------------------
// DPI-related helpers (optional)
//--------------------------------------------------------------------------------------------------------
// - Use to enable DPI awareness without having to create an application manifest.
// - Your own app may already do this via a manifest or explicit calls. This is mostly useful for
// our examples/ apps.
// - In theory we could call simple functions from Windows SDK such as SetProcessDPIAware(),
// SetProcessDpiAwareness(), etc.
//   but most of the functions provided by Microsoft require Windows 8.1/10+ SDK at compile time and
//   Windows 8/10+ at runtime, neither we want to require the user to have. So we dynamically select
//   and load those functions to avoid dependencies.
//---------------------------------------------------------------------------------------------------------
// This is the scheme successfully used by GLFW (from which we borrowed some of the code) and other
// apps aiming to be highly portable. ImGui_ImplWin32_EnableDpiAwareness() is just a helper called
// by main.cpp, we don't call it automatically. If you are trying to implement your own back-end for
// your own engine, you may ignore that noise.
//---------------------------------------------------------------------------------------------------------

// Implement some of the functions and types normally declared in recent Windows SDK.
#  if !defined(_versionhelpers_H_INCLUDED_) && !defined(_INC_VERSIONHELPERS)
static BOOL IsWindowsVersionOrGreater(WORD major, WORD minor, WORD sp) {
  OSVERSIONINFOEXW osvi = {sizeof(osvi), major, minor, 0, 0, {0}, sp, 0, 0, 0, 0};
  DWORD            mask = VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR;
  ULONGLONG        cond = ::VerSetConditionMask(0, VER_MAJORVERSION, VER_GREATER_EQUAL);
  cond                  = ::VerSetConditionMask(cond, VER_MINORVERSION, VER_GREATER_EQUAL);
  cond                  = ::VerSetConditionMask(cond, VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);
  return ::VerifyVersionInfoW(&osvi, mask, cond);
}
#    define IsWindows8Point1OrGreater()                                                            \
      IsWindowsVersionOrGreater(HIBYTE(0x0602), LOBYTE(0x0602), 0) // _WIN32_WINNT_WINBLUE
#  endif

#  ifndef DPI_ENUMS_DECLARED
typedef enum {
  PROCESS_DPI_UNAWARE           = 0,
  PROCESS_SYSTEM_DPI_AWARE      = 1,
  PROCESS_PER_MONITOR_DPI_AWARE = 2
} PROCESS_DPI_AWARENESS;
typedef enum {
  MDT_EFFECTIVE_DPI = 0,
  MDT_ANGULAR_DPI   = 1,
  MDT_RAW_DPI       = 2,
  MDT_DEFAULT       = MDT_EFFECTIVE_DPI
} MONITOR_DPI_TYPE;
#  endif
#  ifndef _DPI_AWARENESS_CONTEXTS_
DECLARE_HANDLE(DPI_AWARENESS_CONTEXT);
#    define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE (DPI_AWARENESS_CONTEXT) - 3
#  endif
#  ifndef DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
#    define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (DPI_AWARENESS_CONTEXT) - 4
#  endif
typedef HRESULT(WINAPI *PFN_SetProcessDpiAwareness)(
    PROCESS_DPI_AWARENESS); // Shcore.lib + dll, Windows 8.1+
typedef HRESULT(WINAPI *PFN_GetDpiForMonitor)(HMONITOR, MONITOR_DPI_TYPE, UINT *,
                                              UINT *); // Shcore.lib + dll, Windows 8.1+
typedef DPI_AWARENESS_CONTEXT(WINAPI *PFN_SetThreadDpiAwarenessContext)(
    DPI_AWARENESS_CONTEXT); // User32.lib + dll, Windows 10 v1607+ (Creators Update)

// Helper function to enable DPI awareness without setting up a manifest
void ImGui_ImplWin32_EnableDpiAwareness() {
  // Make sure monitors will be updated with latest correct scaling
  g_WantUpdateMonitors = true;

  // if (IsWindows10OrGreater()) // This needs a manifest to succeed. Instead we try to grab the
  // function pointer!
  {
    static HINSTANCE user32_dll = ::LoadLibraryA("user32.dll"); // Reference counted per-process
    if (PFN_SetThreadDpiAwarenessContext SetThreadDpiAwarenessContextFn =
            (PFN_SetThreadDpiAwarenessContext)::GetProcAddress(user32_dll,
                                                               "SetThreadDpiAwarenessContext")) {
      SetThreadDpiAwarenessContextFn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
      return;
    }
  }
  if (IsWindows8Point1OrGreater()) {
    static HINSTANCE shcore_dll = ::LoadLibraryA("shcore.dll"); // Reference counted per-process
    if (PFN_SetProcessDpiAwareness SetProcessDpiAwarenessFn =
            (PFN_SetProcessDpiAwareness)::GetProcAddress(shcore_dll, "SetProcessDpiAwareness")) {
      SetProcessDpiAwarenessFn(PROCESS_PER_MONITOR_DPI_AWARE);
      return;
    }
  }
#  if _WIN32_WINNT >= 0x0600
  ::SetProcessDPIAware();
#  endif
}

#  if defined(_MSC_VER) && !defined(NOGDI)
#    pragma comment(lib, "gdi32") // Link with gdi32.lib for GetDeviceCaps()
#  endif

float ImGui_ImplWin32_GetDpiScaleForMonitor(void *monitor) {
  UINT        xdpi = 96, ydpi = 96;
  static BOOL bIsWindows8Point1OrGreater = IsWindows8Point1OrGreater();
  if (bIsWindows8Point1OrGreater) {
    static HINSTANCE shcore_dll = ::LoadLibraryA("shcore.dll"); // Reference counted per-process
    if (PFN_GetDpiForMonitor GetDpiForMonitorFn =
            (PFN_GetDpiForMonitor)::GetProcAddress(shcore_dll, "GetDpiForMonitor"))
      GetDpiForMonitorFn((HMONITOR)monitor, MDT_EFFECTIVE_DPI, &xdpi, &ydpi);
  }
#  ifndef NOGDI
  else {
    const HDC dc = ::GetDC(NULL);
    xdpi         = ::GetDeviceCaps(dc, LOGPIXELSX);
    ydpi         = ::GetDeviceCaps(dc, LOGPIXELSY);
    ::ReleaseDC(NULL, dc);
  }
#  endif
  IM_ASSERT(xdpi == ydpi); // Please contact me if you hit this assert!
  return xdpi / 96.0f;
}

float ImGui_ImplWin32_GetDpiScaleForHwnd(void *hwnd) {
  HMONITOR monitor = ::MonitorFromWindow((HWND)hwnd, MONITOR_DEFAULTTONEAREST);
  return ImGui_ImplWin32_GetDpiScaleForMonitor(monitor);
}

//--------------------------------------------------------------------------------------------------------
// IME (Input Method Editor) basic support for e.g. Asian language users
//--------------------------------------------------------------------------------------------------------

#  if defined(WINAPI_FAMILY) &&                                                                    \
      (WINAPI_FAMILY == WINAPI_FAMILY_APP) // UWP doesn't have Win32 functions
#    define IMGUI_DISABLE_WIN32_DEFAULT_IME_FUNCTIONS
#  endif

#  if defined(_WIN32) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS) &&                                \
      !defined(IMGUI_DISABLE_WIN32_DEFAULT_IME_FUNCTIONS) && !defined(__GNUC__)
#    define HAS_WIN32_IME 1
#    include <imm.h>
#    ifdef _MSC_VER
#      pragma comment(lib, "imm32")
#    endif
static void ImGui_ImplWin32_SetImeInputPos(ImGuiViewport *viewport, ImVec2 pos) {
  COMPOSITIONFORM cf = {CFS_FORCE_POSITION,
                        {(LONG)(pos.x - viewport->Pos.x), (LONG)(pos.y - viewport->Pos.y)},
                        {0, 0, 0, 0}};
  if (HWND hwnd = (HWND)viewport->PlatformHandle)
    if (HIMC himc = ::ImmGetContext(hwnd)) {
      ::ImmSetCompositionWindow(himc, &cf);
      ::ImmReleaseContext(hwnd, himc);
    }
}
#  else
#    define HAS_WIN32_IME 0
#  endif

//--------------------------------------------------------------------------------------------------------
// MULTI-VIEWPORT / PLATFORM INTERFACE SUPPORT
// This is an _advanced_ and _optional_ feature, allowing the back-end to create and handle multiple
// viewports simultaneously. If you are new to dear imgui or creating a new binding for dear imgui,
// it is recommended that you completely ignore this section first..
//--------------------------------------------------------------------------------------------------------

// Helper structure we store in the void* RenderUserData field of each ImGuiViewport to easily
// retrieve our backend data.
struct ImGuiViewportDataWin32 {
  HWND  Hwnd;
  bool  HwndOwned;
  DWORD DwStyle;
  DWORD DwExStyle;

  ImGuiViewportDataWin32() {
    Hwnd      = NULL;
    HwndOwned = false;
    DwStyle = DwExStyle = 0;
  }
  ~ImGuiViewportDataWin32() { IM_ASSERT(Hwnd == NULL); }
};

static void ImGui_ImplWin32_GetWin32StyleFromViewportFlags(ImGuiViewportFlags flags,
                                                           DWORD *out_style, DWORD *out_ex_style) {
  if (flags & ImGuiViewportFlags_NoDecoration)
    *out_style = WS_POPUP;
  else
    *out_style = WS_OVERLAPPEDWINDOW;

  if (flags & ImGuiViewportFlags_NoTaskBarIcon)
    *out_ex_style = WS_EX_TOOLWINDOW;
  else
    *out_ex_style = WS_EX_APPWINDOW;

  if (flags & ImGuiViewportFlags_TopMost) *out_ex_style |= WS_EX_TOPMOST;
}

static void ImGui_ImplWin32_CreateWindow(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = IM_NEW(ImGuiViewportDataWin32)();
  viewport->PlatformUserData   = data;

  // Select style and parent window
  ImGui_ImplWin32_GetWin32StyleFromViewportFlags(viewport->Flags, &data->DwStyle, &data->DwExStyle);
  HWND parent_window = NULL;
  if (viewport->ParentViewportId != 0)
    if (ImGuiViewport *parent_viewport = ImGui::FindViewportByID(viewport->ParentViewportId))
      parent_window = (HWND)parent_viewport->PlatformHandle;

  // Create window
  RECT rect = {(LONG)viewport->Pos.x, (LONG)viewport->Pos.y,
               (LONG)(viewport->Pos.x + viewport->Size.x),
               (LONG)(viewport->Pos.y + viewport->Size.y)};
  ::AdjustWindowRectEx(&rect, data->DwStyle, FALSE, data->DwExStyle);
  data->Hwnd = ::CreateWindowEx(
      data->DwExStyle, _T("ImGui Platform"), _T("Untitled"),
      data->DwStyle, // Style, class name, window name
      rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, // Window area
      parent_window, NULL, ::GetModuleHandle(NULL), NULL); // Parent window, Menu, Instance, Param
  data->HwndOwned                 = true;
  viewport->PlatformRequestResize = false;
  viewport->PlatformHandle = viewport->PlatformHandleRaw = data->Hwnd;
}

static void ImGui_ImplWin32_DestroyWindow(ImGuiViewport *viewport) {
  if (ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData) {
    if (::GetCapture() == data->Hwnd) {
      // Transfer capture so if we started dragging from a window that later disappears, we'll still
      // receive the MOUSEUP event.
      ::ReleaseCapture();
      ::SetCapture(g_hWnd);
    }
    if (data->Hwnd && data->HwndOwned) ::DestroyWindow(data->Hwnd);
    data->Hwnd = NULL;
    IM_DELETE(data);
  }
  viewport->PlatformUserData = viewport->PlatformHandle = NULL;
}

static void ImGui_ImplWin32_ShowWindow(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  if (viewport->Flags & ImGuiViewportFlags_NoFocusOnAppearing)
    ::ShowWindow(data->Hwnd, SW_SHOWNA);
  else
    ::ShowWindow(data->Hwnd, SW_SHOW);
}

static void ImGui_ImplWin32_UpdateWindow(ImGuiViewport *viewport) {
  // (Optional) Update Win32 style if it changed _after_ creation.
  // Generally they won't change unless configuration flags are changed, but advanced uses (such as
  // manually rewriting viewport flags) make this useful.
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  DWORD new_style;
  DWORD new_ex_style;
  ImGui_ImplWin32_GetWin32StyleFromViewportFlags(viewport->Flags, &new_style, &new_ex_style);

  // Only reapply the flags that have been changed from our point of view (as other flags are being
  // modified by Windows)
  if (data->DwStyle != new_style || data->DwExStyle != new_ex_style) {
    data->DwStyle   = new_style;
    data->DwExStyle = new_ex_style;
    ::SetWindowLong(data->Hwnd, GWL_STYLE, data->DwStyle);
    ::SetWindowLong(data->Hwnd, GWL_EXSTYLE, data->DwExStyle);
    RECT rect = {(LONG)viewport->Pos.x, (LONG)viewport->Pos.y,
                 (LONG)(viewport->Pos.x + viewport->Size.x),
                 (LONG)(viewport->Pos.y + viewport->Size.y)};
    ::AdjustWindowRectEx(&rect, data->DwStyle, FALSE, data->DwExStyle); // Client to Screen
    ::SetWindowPos(data->Hwnd, NULL, rect.left, rect.top, rect.right - rect.left,
                   rect.bottom - rect.top, SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    ::ShowWindow(data->Hwnd, SW_SHOWNA); // This is necessary when we alter the style
    viewport->PlatformRequestMove = viewport->PlatformRequestResize = true;
  }
}

static ImVec2 ImGui_ImplWin32_GetWindowPos(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  POINT pos = {0, 0};
  ::ClientToScreen(data->Hwnd, &pos);
  return ImVec2((float)pos.x, (float)pos.y);
}

static void ImGui_ImplWin32_SetWindowPos(ImGuiViewport *viewport, ImVec2 pos) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  RECT rect = {(LONG)pos.x, (LONG)pos.y, (LONG)pos.x, (LONG)pos.y};
  ::AdjustWindowRectEx(&rect, data->DwStyle, FALSE, data->DwExStyle);
  ::SetWindowPos(data->Hwnd, NULL, rect.left, rect.top, 0, 0,
                 SWP_NOZORDER | SWP_NOSIZE | SWP_NOACTIVATE);
}

static ImVec2 ImGui_ImplWin32_GetWindowSize(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  RECT rect;
  ::GetClientRect(data->Hwnd, &rect);
  return ImVec2(float(rect.right - rect.left), float(rect.bottom - rect.top));
}

static void ImGui_ImplWin32_SetWindowSize(ImGuiViewport *viewport, ImVec2 size) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  RECT rect = {0, 0, (LONG)size.x, (LONG)size.y};
  ::AdjustWindowRectEx(&rect, data->DwStyle, FALSE, data->DwExStyle); // Client to Screen
  ::SetWindowPos(data->Hwnd, NULL, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOZORDER | SWP_NOMOVE | SWP_NOACTIVATE);
}

static void ImGui_ImplWin32_SetWindowFocus(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  ::BringWindowToTop(data->Hwnd);
  ::SetForegroundWindow(data->Hwnd);
  ::SetFocus(data->Hwnd);
}

static bool ImGui_ImplWin32_GetWindowFocus(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  return ::GetForegroundWindow() == data->Hwnd;
}

static bool ImGui_ImplWin32_GetWindowMinimized(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  return ::IsIconic(data->Hwnd) != 0;
}

static void ImGui_ImplWin32_SetWindowTitle(ImGuiViewport *viewport, const char *title) {
  // ::SetWindowTextA() doesn't properly handle UTF-8 so we explicitely convert our string.
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  int               n = ::MultiByteToWideChar(CP_UTF8, 0, title, -1, NULL, 0);
  ImVector<wchar_t> title_w;
  title_w.resize(n);
  ::MultiByteToWideChar(CP_UTF8, 0, title, -1, title_w.Data, n);
  ::SetWindowTextW(data->Hwnd, title_w.Data);
}

static void ImGui_ImplWin32_SetWindowAlpha(ImGuiViewport *viewport, float alpha) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  IM_ASSERT(alpha >= 0.0f && alpha <= 1.0f);
  if (alpha < 1.0f) {
    DWORD style = ::GetWindowLongW(data->Hwnd, GWL_EXSTYLE) | WS_EX_LAYERED;
    ::SetWindowLongW(data->Hwnd, GWL_EXSTYLE, style);
    ::SetLayeredWindowAttributes(data->Hwnd, 0, (BYTE)(255 * alpha), LWA_ALPHA);
  } else {
    DWORD style = ::GetWindowLongW(data->Hwnd, GWL_EXSTYLE) & ~WS_EX_LAYERED;
    ::SetWindowLongW(data->Hwnd, GWL_EXSTYLE, style);
  }
}

static float ImGui_ImplWin32_GetWindowDpiScale(ImGuiViewport *viewport) {
  ImGuiViewportDataWin32 *data = (ImGuiViewportDataWin32 *)viewport->PlatformUserData;
  IM_ASSERT(data->Hwnd != 0);
  return ImGui_ImplWin32_GetDpiScaleForHwnd(data->Hwnd);
}

// FIXME-DPI: Testing DPI related ideas
static void ImGui_ImplWin32_OnChangedViewport(ImGuiViewport *viewport) {
  (void)viewport;
#  if 0
    ImGuiStyle default_style;
    //default_style.WindowPadding = ImVec2(0, 0);
    //default_style.WindowBorderSize = 0.0f;
    //default_style.ItemSpacing.y = 3.0f;
    //default_style.FramePadding = ImVec2(0, 0);
    default_style.ScaleAllSizes(viewport->DpiScale);
    ImGuiStyle& style = ImGui::GetStyle();
    style = default_style;
#  endif
}

static LRESULT CALLBACK ImGui_ImplWin32_WndProcHandler_PlatformWindow(HWND hWnd, UINT msg,
                                                                      WPARAM wParam,
                                                                      LPARAM lParam) {
  if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;

  if (ImGuiViewport *viewport = ImGui::FindViewportByPlatformHandle((void *)hWnd)) {
    switch (msg) {
    case WM_CLOSE: viewport->PlatformRequestClose = true; return 0;
    case WM_MOVE: viewport->PlatformRequestMove = true; break;
    case WM_SIZE: viewport->PlatformRequestResize = true; break;
    case WM_MOUSEACTIVATE:
      if (viewport->Flags & ImGuiViewportFlags_NoFocusOnClick) return MA_NOACTIVATE;
      break;
    case WM_NCHITTEST:
      // Let mouse pass-through the window. This will allow the back-end to set
      // io.MouseHoveredViewport properly (which is OPTIONAL). The ImGuiViewportFlags_NoInputs flag
      // is set while dragging a viewport, as want to detect the window behind the one we are
      // dragging. If you cannot easily access those viewport flags from your windowing/event code:
      // you may manually synchronize its state e.g. in your main loop after calling
      // UpdatePlatformWindows(). Iterate all viewports/platform windows and pass the flag to your
      // windowing system.
      if (viewport->Flags & ImGuiViewportFlags_NoInputs) return HTTRANSPARENT;
      break;
    }
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}

static void ImGui_ImplWin32_InitPlatformInterface() {
  WNDCLASSEX wcex;
  wcex.cbSize        = sizeof(WNDCLASSEX);
  wcex.style         = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc   = ImGui_ImplWin32_WndProcHandler_PlatformWindow;
  wcex.cbClsExtra    = 0;
  wcex.cbWndExtra    = 0;
  wcex.hInstance     = ::GetModuleHandle(NULL);
  wcex.hIcon         = NULL;
  wcex.hCursor       = NULL;
  wcex.hbrBackground = (HBRUSH)(COLOR_BACKGROUND + 1);
  wcex.lpszMenuName  = NULL;
  wcex.lpszClassName = _T("ImGui Platform");
  wcex.hIconSm       = NULL;
  ::RegisterClassEx(&wcex);

  ImGui_ImplWin32_UpdateMonitors();

  // Register platform interface (will be coupled with a renderer interface)
  ImGuiPlatformIO &platform_io            = ImGui::GetPlatformIO();
  platform_io.Platform_CreateWindow       = ImGui_ImplWin32_CreateWindow;
  platform_io.Platform_DestroyWindow      = ImGui_ImplWin32_DestroyWindow;
  platform_io.Platform_ShowWindow         = ImGui_ImplWin32_ShowWindow;
  platform_io.Platform_SetWindowPos       = ImGui_ImplWin32_SetWindowPos;
  platform_io.Platform_GetWindowPos       = ImGui_ImplWin32_GetWindowPos;
  platform_io.Platform_SetWindowSize      = ImGui_ImplWin32_SetWindowSize;
  platform_io.Platform_GetWindowSize      = ImGui_ImplWin32_GetWindowSize;
  platform_io.Platform_SetWindowFocus     = ImGui_ImplWin32_SetWindowFocus;
  platform_io.Platform_GetWindowFocus     = ImGui_ImplWin32_GetWindowFocus;
  platform_io.Platform_GetWindowMinimized = ImGui_ImplWin32_GetWindowMinimized;
  platform_io.Platform_SetWindowTitle     = ImGui_ImplWin32_SetWindowTitle;
  platform_io.Platform_SetWindowAlpha     = ImGui_ImplWin32_SetWindowAlpha;
  platform_io.Platform_UpdateWindow       = ImGui_ImplWin32_UpdateWindow;
  platform_io.Platform_GetWindowDpiScale  = ImGui_ImplWin32_GetWindowDpiScale; // FIXME-DPI
  platform_io.Platform_OnChangedViewport  = ImGui_ImplWin32_OnChangedViewport; // FIXME-DPI
#  if HAS_WIN32_IME
  platform_io.Platform_SetImeInputPos     = ImGui_ImplWin32_SetImeInputPos;
#  endif

  // Register main window handle (which is owned by the main application, not by us)
  // This is mostly for simplicity and consistency, so that our code (e.g. mouse handling etc.) can
  // use same logic for main and secondary viewports.
  ImGuiViewport *         main_viewport = ImGui::GetMainViewport();
  ImGuiViewportDataWin32 *data          = IM_NEW(ImGuiViewportDataWin32)();
  data->Hwnd                            = g_hWnd;
  data->HwndOwned                       = false;
  main_viewport->PlatformUserData       = data;
  main_viewport->PlatformHandle         = (void *)g_hWnd;
}

static void ImGui_ImplWin32_ShutdownPlatformInterface() {
  ::UnregisterClass(_T("ImGui Platform"), ::GetModuleHandle(NULL));
}

//---------------------------------------------------------------------------------------------------------
#endif