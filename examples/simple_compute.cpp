#include "rendering.hpp"
#include "rendering_utils.hpp"

#ifdef __linux__
#  include <SDL2/SDL.h>
#else
#  include <SDL.h>
#endif

class Event_Consumer : public rd::IEvent_Consumer {
  public:
  void consume(void *_event) override {}
  void on_frame(rd::IFactory *factory) override {
    auto                  sc_info = factory->get_swapchain_image_info();
    rd::Image_Create_Info ci;
    MEMZERO(ci);
    ci.format     = rd::Format::RGBA32_FLOAT;
    ci.depth      = 1;
    ci.width      = sc_info.width;
    ci.height     = sc_info.height;
    ci.layers     = 1;
    ci.levels     = 1;
    ci.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
    ci.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV | //
                    (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
    Resource_ID img = factory->create_image(ci);
    defer(factory->release_resource(img));
    {
      rd::Imm_Ctx *ctx = factory->start_compute_pass();
      ctx->CS_set_shader(factory->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 0)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  float2 uv = float2(GLOBAL_THREAD_INDEX.xy) / dim.xy;
  image_store(out_image, GLOBAL_THREAD_INDEX.xy, float4(uv, 0.0, 1.0));
@(END)
)"),
                                                    NULL, 0));
      ctx->bind_rw_image(0, 0, 0, img, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
      ctx->image_barrier(img, (u32)rd::Access_Bits::MEMORY_WRITE,
                         rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
      ctx->dispatch((sc_info.width + 15) / 16, (sc_info.height + 15) / 16, 1);
      factory->end_compute_pass(ctx);
    }
    {
      rd::Render_Pass_Create_Info info;
      MEMZERO(info);
      rd::RT_View rt0;
      MEMZERO(rt0);
      rt0.image             = factory->get_swapchain_image();
      rt0.format            = rd::Format::NATIVE;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 1.0f;
      info.rts.push(rt0);
      rd::Imm_Ctx *ctx = factory->start_render_pass(info);
      ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_OUTPUT (location 0) (type float2) (name UV))
@(ENTRY)
  float x = -1.0 + float((VERTEX_INDEX & 1) << 2);
  float y = -1.0 + float((VERTEX_INDEX & 2) << 1);
  UV = float2(x, -y) * 0.5 + float2_splat(0.5);
  @(EXPORT_POSITION float4(x, y, 0.0, 1.0));
@(END)
)"),
                                                    NULL, 0));
      ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 0)
  (format RGBA32_FLOAT)
  (name sTexture)
)
@(DECLARE_SAMPLER
  (set 0)
  (binding 1)
  (name sSampler)
)
@(DECLARE_INPUT (location 0) (type float2) (name UV))
@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  @(EXPORT_COLOR 0 texture(sampler2D(sTexture, sSampler), UV));
@(END)
)"),
                                                    NULL, 0));
      rd::Blend_State bs;
      MEMZERO(bs);
      bs.enabled = false;
      bs.color_write_mask =
          (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
          (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
      ito(1) ctx->OM_set_blend_state(i, bs);
      ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
      ctx->RS_set_state(rs_state);
      rd::DS_State ds_state;
      MEMZERO(ds_state);
      ds_state.cmp_op             = rd::Cmp::EQ;
      ds_state.enable_depth_test  = false;
      ds_state.enable_depth_write = false;
      ctx->DS_set_state(ds_state);
      rd::MS_State ms_state;
      MEMZERO(ms_state);
      ms_state.sample_mask = 0xffffffffu;
      ms_state.num_samples = 1;
      ctx->MS_set_state(ms_state);

      Resource_ID sampler;
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
        sampler             = factory->create_sampler(info);
      }
      defer(factory->release_resource(sampler));
      ctx->bind_sampler(0, 1, sampler);
      ctx->bind_image(0, 0, 0, img, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
      ctx->set_viewport(0.0f, 0.0f, (float)sc_info.width, (float)sc_info.height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, sc_info.width, sc_info.height);
      ctx->draw(3, 1, 0, 0);
      factory->end_render_pass(ctx);
    }
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  IGUI_Pass *gui = new IGUI_Pass;
  gui->init(pmng);
  pmng->set_event_consumer(gui);
  pmng->loop();
  return 0;
}
