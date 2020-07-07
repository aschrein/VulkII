#include "rendering.hpp"
#include "script.hpp"

class Simple_Pass : public rd::IPass {
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
    // rd::Image2D_Create_Info rt0_info;
    // MEMZERO(rt0_info);
    // rt0_info.format     = rd::Format::RGBA32_FLOAT;
    // rt0_info.width      = 512;
    // rt0_info.height     = 512;
    // rt0_info.layers     = 1;
    // rt0_info.levels     = 1;
    // rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE;
    // rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |
    //                      (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
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

    vs = pc->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(OUTPUT 0 float2 tex_coords);
@(ENTRY)
  float x = -1.0 + float((VERTEX_INDEX & 1) << 2);
  float y = -1.0 + float((VERTEX_INDEX & 2) << 1);
  tex_coords = float2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  @(EXPORT_POSITION
      float4(x, (y + 0.0), 0.5, 1.0)
  );
@(END)
)"),
                               NULL, 0);
    ps = pc->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
layout(set = 0, binding = 0, std140) uniform UBO {
  float4 color;
} g_ubo;
@(DECLARE_RENDER_TARGET 0);
@(ENTRY)
  @(EXPORT_COLOR 0 float4(g_ubo.color.xyz, 1.0));
@(END)
)"),
                               NULL, 0);
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
    buf_info.size = 16;
    uniform_buffer = pc->create_buffer(buf_info);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    rd::Blend_State bs;
    MEMZERO(bs);
    bs.enabled          = false;
    bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT |
                          (u32)rd::Color_Component_Bit::G_BIT |
                          (u32)rd::Color_Component_Bit::B_BIT |
                          (u32)rd::Color_Component_Bit::A_BIT;
    ctx->OM_set_blend_state(0, bs);
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
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width / 2, height);
    {
      float *ptr = (float*)ctx->map_buffer(uniform_buffer);
      ptr[0] = 1.0f;
      ptr[1] = 1.0f;
      ptr[2] = 0.0f;
      ptr[3] = 1.0f;
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(rd::Stage_t::PIXEL, 0, 0, uniform_buffer, 0, 16);
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
  // ASSERT_ALWAYS(argc == 2);
  // IEvaluator::parse_and_eval(stref_s(read_file_tmp(argv[1])));
  Simple_Pass   simple_pass;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  pmng->add_pass(&simple_pass);
  pmng->loop();
  return 0;
}
