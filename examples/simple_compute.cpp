
#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <atomic>
//#include <functional>
#include <3rdparty/half.hpp>
#include <imgui.h>
#include <mutex>
#include <thread>

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
    rd::Imm_Ctx *ctx = factory->start_compute_pass();
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
    ctx->dispatch(1, 1, 1);
    factory->end_compute_pass(ctx);
    factory->wait_idle();
    u32 *map = (u32 *)factory->map_buffer(res_id);
    ito(1024)
      fprintf(stdout, "%i ", map[i]);
    factory->unmap_buffer(res_id);
    factory->release_resource(res_id);
  }
  void on_release(rd::IFactory *factory) override {}
  void on_frame(rd::IFactory *factory) override {}
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  IGUI_Pass *   gui  = new Event_Consumer;
  gui->init(pmng);
  pmng->set_event_consumer(gui);
  pmng->loop();
  return 0;
}
