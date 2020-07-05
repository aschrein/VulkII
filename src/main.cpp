#include "rendering.hpp"
#define SCRIPT_IMPL
#include "script.hpp"

String_Builder sb;

string_ref preprocess(char const *body) {
  sb.reset();
  sb.putf("#version 450\n");
  sb.putf("#extension GL_EXT_nonuniform_qualifier : require\n");
  sb.putf(R"(
#define float2        vec2
#define float3        vec3
#define float4        vec4
#define int2          ivec2
#define int3          ivec3
#define int4          ivec4
#define uint2         uvec2
#define uint3         uvec3
#define uint4         uvec4
#define float2x2      mat2
#define float3x3      mat3
#define float4x4      mat4
#define VERTEX_INDEX  gl_VertexIndex

)");
  sb.putf(body);
  return sb.get_str();
}

class Simple_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;

  public:
  void on_end(rd::IResource_Manager *rm) override {}
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

    //rd::Clear_Color cl;
    //cl.clear = true;
    //cl.r     = 0.0f;
    //cl.g     = 0.0f;
    //cl.b     = 0.0f;
    //cl.a     = 0.0f;
    //pc->add_render_target(pc->get_swapchain_image(), 0, 0, cl);

    vs = pc->create_shader_raw(rd::Stage_t::VERTEX, preprocess(R"(\
layout(location = 0) out vec2 tex_coords;
void main() {
  float x = -1.0 + float((gl_VertexIndex & 1) << 2);
  float y = -1.0 + float((gl_VertexIndex & 2) << 1);
  tex_coords = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  gl_Position = vec4(x, y, 1, 1);
}
)"),
                               NULL, 0);
    ps = pc->create_shader_raw(rd::Stage_t::PIXEL, preprocess(R"(\
layout(location = 0) out vec4 f_color;
void main() { f_color = vec4(1.0, 0.0, 0.0, 1.0); }
)"),
                               NULL, 0);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    rd::Blend_State bs;
    MEMZERO(bs);
    bs.enabled = false;
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
    ms_state.num_samples = 1;
    ctx->MS_set_state(ms_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
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
  sb.init();
  // ASSERT_ALWAYS(argc == 2);
  // IEvaluator::parse_and_eval(stref_s(read_file_tmp(argv[1])));
  Simple_Pass   simple_pass;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  pmng->add_pass(&simple_pass);
  pmng->loop();
  return 0;
}
