#include "rendering.hpp"
#include "script.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace glm;

using int2     = ivec2;
using int3     = ivec3;
using int4     = ivec4;
using uint2    = uvec2;
using uint3    = uvec3;
using uint4    = uvec4;
using float2   = vec2;
using float3   = vec3;
using float4   = vec4;
using float2x2 = mat2;
using float3x3 = mat3;
using float4x4 = mat4;

static void setup_default_state(rd::Imm_Ctx *ctx, u32 num_rts = 1) {
  rd::Blend_State bs;
  MEMZERO(bs);
  bs.enabled          = false;
  bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT |
                        (u32)rd::Color_Component_Bit::G_BIT |
                        (u32)rd::Color_Component_Bit::B_BIT |
                        (u32)rd::Color_Component_Bit::A_BIT;
  ito(num_rts) ctx->OM_set_blend_state(i, bs);
  ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
  rd::RS_State rs_state;
  MEMZERO(rs_state);
  rs_state.polygon_mode = rd::Polygon_Mode::FILL;
  rs_state.front_face   = rd::Front_Face::CW;
  rs_state.cull_mode    = rd::Cull_Mode::NONE;
  rs_state.line_width   = 1.0f;
  rs_state.depth_bias   = 0.0f;
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
}

class Opaque_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID uniform_buffer;

  public:
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    rd::Image2D_Info info = pc->get_swapchain_image_info();
    width                 = info.width;
    height                = info.height;

    rd::Clear_Color cl;
    cl.clear = true;
    cl.r     = 0.0f;
    cl.g     = 0.0f;
    cl.b     = 0.0f;
    cl.a     = 0.0f;

    rd::Image_Create_Info rt0_info;
    MEMZERO(rt0_info);
    rt0_info.format     = rd::Format::RGBA32_FLOAT;
    rt0_info.width      = width;
    rt0_info.height     = height;
    rt0_info.depth      = 1;
    rt0_info.layers     = 1;
    rt0_info.levels     = 1;
    rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
    rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
    pc->add_render_target(stref_s("opaque_pass/rt0"), rt0_info, 0, 0, cl);

    vs = pc->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_OUTPUT
  (location 0)
  (type float2)
  (name tex_coords)
)

@(ENTRY)
  float x = -1.0 + float((VERTEX_INDEX & 1) << 2);
  float y = -1.0 + float((VERTEX_INDEX & 2) << 1);
  tex_coords = float2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  x /= 4.0;
  y /= 4.0;
  @(EXPORT_POSITION
      float4(x, y, 0.5, 1.0)
  );
@(END)
)"),
                               NULL, 0);
    ps = pc->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT
  (location 0)
  (type float2)
  (name tex_coords)
)
@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  @(EXPORT_COLOR 0 float4(tex_coords.xy, 0.0, 1.0));
@(END)
)"),
                               NULL, 0);
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
    buf_info.size       = 16;
    uniform_buffer      = pc->create_buffer(buf_info);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    setup_default_state(ctx);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      float *ptr = (float *)ctx->map_buffer(uniform_buffer);
      ptr[0]     = 1.0f;
      ptr[1]     = 1.0f;
      ptr[2]     = 0.0f;
      ptr[3]     = 1.0f;
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(rd::Stage_t::PIXEL, 0, 0, uniform_buffer, 0, 16);
    ctx->flush_bindings();
    ctx->draw(3, 1, 0, 0);
  }
  string_ref get_name() override { return stref_s("opaque_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
  }
};

class Merge_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID uniform_buffer;
  Resource_ID sampler;
  Resource_ID my_image;

  struct Uniform {
    float2x2 rot;
    float4   color;
  } uniform_data;

  public:
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    rd::Image2D_Info info = pc->get_swapchain_image_info();
    width                 = info.width;
    height                = info.height;

    rd::Clear_Color cl;
    cl.clear = true;
    cl.r     = 0.0f;
    cl.g     = 0.0f;
    cl.b     = 0.0f;
    cl.a     = 0.0f;

    pc->add_render_target(pc->get_swapchain_image(), 0, 0, cl);
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float2x2) (name rot))
  (add_field (type float4)   (name color))
)
@(DECLARE_OUTPUT
  (location 0)
  (type float2)
  (name tex_coords)
)

@(ENTRY)
  float x = -1.0 + float((VERTEX_INDEX & 1) << 2);
  float y = -1.0 + float((VERTEX_INDEX & 2) << 1);
  tex_coords = rot * float2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  @(EXPORT_POSITION
      float4(x, y, 0.5, 1.0)
  );
@(END)
)"),
                                 NULL, 0);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 1)
  (binding 0)
  (add_field (type float2x2) (name rot))
  (add_field (type float4)   (name color))
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 1)
  (binding 1)
  (format RGBA32_FLOAT)
  (name my_image)
)
@(DECLARE_SAMPLER
  (set 1)
  (binding 2)
  (name my_sampler)
)
@(DECLARE_INPUT
  (location 0)
  (type float2)
  (name tex_coords)
)
@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  @(EXPORT_COLOR 0
        lerp(
            float4(color.xyz, 1.0),
            texture(sampler2D(my_image, my_sampler), tex_coords),
            0.5)
        );
@(END)
)"),
                                 NULL, 0);
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
    buf_info.size       = sizeof(uniform_data);
    uniform_buffer      = pc->create_buffer(buf_info);
    if (sampler.is_null()) {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      sampler             = pc->create_sampler(info);
    }
    my_image = pc->get_resource(stref_s("opaque_pass/rt0"));
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (my_image.is_null()) return;
    setup_default_state(ctx);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      Uniform *ptr = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->rot     = float2x2( //
          1.0f, 0.0,       //
          0.0, 1.0f        //
      );
      ptr->color   = float4(1.0f, 0.0f, 0.0f, 1.0f);
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(rd::Stage_t::PIXEL, 1, 0, uniform_buffer, 0,
                             sizeof(Uniform));
    ctx->bind_uniform_buffer(rd::Stage_t::VERTEX, 0, 0, uniform_buffer, 0,
                             sizeof(Uniform));
    ctx->bind_image(rd::Stage_t::PIXEL, 1, 1, 0, my_image, 0, 1, 0, 1);
    ctx->bind_sampler(rd::Stage_t::PIXEL, 1, 2, sampler);

    ctx->flush_bindings();
    ctx->draw(3, 1, 0, 0);
  }
  string_ref get_name() override { return stref_s("simple_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  pmng->add_pass(new Opaque_Pass);
  pmng->add_pass(new Merge_Pass);
  pmng->loop();
  return 0;
}
