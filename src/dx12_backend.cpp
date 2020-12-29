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

#if 0
				class Dx12Factory : public rd::IDevice {
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
  bool        get_timestamp_state(Resource_ID t0) override { TRAP; }
  double      get_timestamp_ms(Resource_ID t0, Resource_ID t1) override { TRAP; }
  Resource_ID create_image(rd::Image_Create_Info info) override { TRAP; }
  Resource_ID create_buffer(rd::Buffer_Create_Info info) override { TRAP; }
  bool        get_event_state(Resource_ID fence_id) override { TRAP; }
  Resource_ID create_shader_raw(rd::Stage_t type, string_ref body,
                                Pair<string_ref, string_ref> *defines,
                                size_t                        num_defines) override {

    TRAP;
  }
  void *           map_buffer(Resource_ID res_id) override { TRAP; }
  void             unmap_buffer(Resource_ID res_id) override { TRAP; }
  Resource_ID      create_sampler(rd::Sampler_Create_Info const &info) override { TRAP; }
  void             release_resource(Resource_ID id) override { TRAP; }
  Resource_ID      get_swapchain_image() override { TRAP; }
  rd::Image2D_Info get_swapchain_image_info() override { TRAP; }
  rd::Image_Info   get_image_info(Resource_ID res_id) override { TRAP; }
  rd::ICtx *    start_render_pass(rd::Render_Pass_Create_Info const &info) override { TRAP; }
  void             end_render_pass(rd::ICtx *_ctx) override { TRAP; }
  rd::ICtx *    start_compute_pass() override { TRAP; }
  void             wait_idle() { TRAP; }
  void             end_compute_pass(rd::ICtx *_ctx) override { TRAP; }
};
class Dx12Pass_Mng : public rd::Pass_Mng {
  public:
  Window *             wnd      = NULL;
  Dx12Factory *        f        = NULL;
  rd::IEvent_Consumer *consumer = NULL;
  Dx12Pass_Mng() {
    wnd = new Window();
    wnd->init();
    f = new Dx12Factory();
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
    wnd->end_frame();
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
      f->on_frame_end();
      wnd->end_frame();
    }
  }
  void  set_event_consumer(rd::IEvent_Consumer *consumer) override { this->consumer = consumer; }
  void *get_window_handle() { return (void *)wnd->window; }
};
rd::Pass_Mng *create_dx12_pass_mng() { return new Dx12Pass_Mng; }
#endif // 0

} // namespace