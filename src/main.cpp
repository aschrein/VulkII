#include "rendering.hpp"
#include "script.hpp"

#include "scene.hpp"

struct Camera {
  float  phi;
  float  theta;
  float  distance;
  float  mx;
  float  my;
  float3 look_at;
  float  aspect;
  float  fov;
  float  znear;
  float  zfar;

  float3   pos;
  float4x4 view;
  float4x4 proj;
  float3   look;
  float3   right;
  float3   up;

  void init() {
    phi      = PI / 2.0f;
    theta    = PI / 2.0f;
    distance = 60.0f;
    mx       = 0.0f;
    my       = 0.0f;
    look_at  = float3(0.0f, 0.0f, 0.0f);
    aspect   = 1.0;
    fov      = PI / 2.0;
    znear    = 1.0e-3f;
    zfar     = 10.0e5f;
  }

  void release() {}

  void update(float2 jitter = float2(0.0f, 0.0f)) {
    pos = float3(sinf(theta) * cosf(phi), cos(theta), sinf(theta) * sinf(phi)) *
              distance +
          look_at;
    look              = normalize(look_at - pos);
    right             = normalize(cross(look, float3(0.0f, 1.0f, 0.0f)));
    up                = normalize(cross(right, look));
    proj              = float4x4(0.0f);
    float tanHalfFovy = std::tan(fov * 0.5f);

    proj[0][0] = 1.0f / (aspect * tanHalfFovy);
    proj[1][1] = 1.0f / (tanHalfFovy);
    proj[2][2] = 0.0f;
    proj[2][3] = -1.0f;
    proj[3][2] = znear;

    proj[2][0] += jitter.x;
    proj[2][1] += jitter.x;
    view = glm::lookAt(pos, look_at, float3(0.0f, 1.0f, 0.0f));
  }
  float4x4 viewproj() { return proj * view; }
};

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

struct GPUMesh {
  PBR_Model * model;
  u32         mesh_index;
  Resource_ID vertex_buffer;
  Resource_ID index_buffer;
  bool        initialized;
  bool        data_uploaded;
  Timer       timer;

  struct Push_Constants {
    afloat3   min;
    afloat3   max;
    afloat4x4 model;
  };

  static_assert(sizeof(Push_Constants) == 96, "Uniform packing is wrong");

  void init(PBR_Model *model, u32 mesh_index) {
    memset(this, 0, sizeof(*this));
    initialized      = false;
    data_uploaded    = false;
    this->model      = model;
    this->mesh_index = mesh_index;
    timer.init();
  }
  void update_buffers(rd::IResource_Manager *rm) {
    initialized = true;
    if (vertex_buffer.is_null()) {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size       = model->meshes[mesh_index].attribute_data.size;
      vertex_buffer       = rm->create_buffer(buf_info);
    }
    if (index_buffer.is_null()) {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size       = model->meshes[mesh_index].index_data.size;
      index_buffer        = rm->create_buffer(buf_info);
    }
  }
  void draw(rd::Imm_Ctx *ctx) {
    timer.update();
    Raw_Mesh_Opaque &mesh = model->meshes[mesh_index];
    if (!data_uploaded) {
      {
        float3 *ptr = (float3 *)(ctx->map_buffer(vertex_buffer));
        memcpy(ptr, &mesh.attribute_data[0], mesh.attribute_data.size);
        ctx->unmap_buffer(vertex_buffer);
      }
      {
        u32 *ptr = (u32 *)ctx->map_buffer(index_buffer);
        u8 * src = &mesh.index_data[0];
        memcpy(ptr, src, mesh.index_data.size);
        ctx->unmap_buffer(index_buffer);
      }
      data_uploaded = true;
    }
    {
      Push_Constants pc;
      MEMZERO(pc);
      pc.min   = mesh.min;
      pc.max   = mesh.max;
      pc.model = rotate(float4x4(1.0f), (float)timer.cur_time, float3(0.0f, 1.0f, 0.0f)) *
                 scale(float4x4(1.0f), float3(1.0f, -1.0f, 1.0f));
      ctx->push_constants(&pc, sizeof(pc));
    }
    Attribute pos_attr = mesh.get_attribute(rd::Attriute_t::POSITION);
    Attribute uv0_attr = mesh.get_attribute(rd::Attriute_t::TEXCOORD0);
    ctx->IA_set_vertex_buffer(0, vertex_buffer, pos_attr.offset, 12,
                              rd::Input_Rate::VERTEX);
    ctx->IA_set_vertex_buffer(1, vertex_buffer, uv0_attr.offset, 8,
                              rd::Input_Rate::VERTEX);
    ctx->IA_set_index_buffer(index_buffer, 0, mesh.index_type);
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGBA32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 1;
      info.offset   = 0;
      info.type     = rd::Attriute_t::TEXCOORD0;
      ctx->IA_set_attribute(info);
    }
    ctx->draw_indexed(mesh.num_indices, 1, 0, 0, 0);
  }
  void release(rd::IResource_Manager *rm) {
    rm->release_resource(vertex_buffer);
    rm->release_resource(index_buffer);
  }
};

class Opaque_Pass : public rd::IPass {
  Resource_ID    vs;
  Resource_ID    ps;
  u32            width, height;
  Resource_ID    uniform_buffer;
  Array<GPUMesh> gpu_meshes;
  PBR_Model      model;
  Camera         camera;

  struct Uniform {
    afloat4x4 viewproj;
  };

  static_assert(sizeof(Uniform) == 64, "Uniform packing is wrong");

  public:
  Opaque_Pass() {
    gpu_meshes.init();
    model = load_gltf_pbr(stref_s("models/old_tree/scene.gltf"));

    gpu_meshes.resize(model.meshes.size);
    ito(model.meshes.size) { gpu_meshes[i].init(&model, i); }

    camera.init();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {

    ito(model.meshes.size) { gpu_meshes[i].update_buffers(pc); }

    rd::Image2D_Info info = pc->get_swapchain_image_info();
    width                 = info.width;
    height                = info.height;
    camera.aspect         = (float)width / height;
    camera.update();
    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.0f;
      cl.g     = 0.0f;
      cl.b     = 1.0f;
      cl.a     = 1.0f;
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
    }
    {
      rd::Clear_Depth cl;
      cl.clear = true;
      cl.d     = 0.0f;
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::D32_FLOAT;
      info.width      = width;
      info.height     = height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      pc->add_depth_target(stref_s("opaque_pass/ds"), info, 0, 0, cl);
    }
    vs = pc->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4) (name viewproj))
)
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float3)   (name mesh_min))
  (add_field (type float3)   (name mesh_max))
  (add_field (type float4x4) (name world_matrix))
)
@(DECLARE_INPUT
  (location 0)
  (type float3)
  (name vertex_position)
)
@(DECLARE_INPUT
  (location 1)
  (type float2)
  (name vertex_uv0)
)
@(DECLARE_OUTPUT
  (location 0)
  (type float2)
  (name tex_coords)
)

@(ENTRY)
  tex_coords = vertex_uv0;
  float3 dsize = mesh_max - mesh_min;
  float scale = max(dsize.x, max(dsize.y, dsize.z));
  @(EXPORT_POSITION
      viewproj * world_matrix * float4(vertex_position * 50.0 / scale, 1.0)
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
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = sizeof(Uniform);
      uniform_buffer      = pc->create_buffer(buf_info);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    setup_default_state(ctx);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      Uniform *ptr = (Uniform *)ctx->map_buffer(uniform_buffer);

      ptr->viewproj = camera.viewproj();
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ito(model.meshes.size) { gpu_meshes[i].draw(ctx); }
  }
  string_ref get_name() override { return stref_s("opaque_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    ito(model.meshes.size) { gpu_meshes[i].release(rm); }
    rm->release_resource(vs);
    rm->release_resource(ps);
    gpu_meshes.release();
  }
};

class Merge_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID uniform_buffer;
  Resource_ID sampler;
  Resource_ID my_image;
  Timer       timer;

  struct Uniform {
    float2x2 rot;
    float4   color;
  } uniform_data;

  public:
  Merge_Pass() { timer.init(); }
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
    static string_ref            shader    = stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4)   (name rot))
  (add_field (type float4)   (name color))
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name my_image)
)
@(DECLARE_SAMPLER
  (set 0)
  (binding 2)
  (name my_sampler)
)
#ifdef VERTEX
@(DECLARE_OUTPUT
  (location 0)
  (type float2)
  (name tex_coords)
)

@(ENTRY)
  float x = -1.0 + float((VERTEX_INDEX & 1) << 2);
  float y = -1.0 + float((VERTEX_INDEX & 2) << 1);
  float2 pos =float2(x, y);
  tex_coords = pos * 0.5 + float2(0.5, 0.5);
  @(EXPORT_POSITION
      float4(pos.xy, 0.5, 1.0)
  );
@(END)
#endif
#ifdef PIXEL
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
            color.a)
        );
@(END)
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);

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
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      sampler             = pc->create_sampler(info);
    }
    my_image = pc->get_resource(stref_s("opaque_pass/rt0"));
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (my_image.is_null()) return;
    timer.update();
    setup_default_state(ctx);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      Uniform *ptr = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->rot     = float2x2(                                    //
          std::cos(timer.cur_time), std::sin(timer.cur_time), //
          -std::sin(timer.cur_time), std::cos(timer.cur_time) //
      );
      ptr->color   = float4(0.5f + 0.5f * std::cos(timer.cur_time), 0.0f, 0.0f,
                          0.5f + 0.5f * std::cos(timer.cur_time));
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_image(0, 1, 0, my_image, 0, 1, 0, 1);
    ctx->bind_sampler(0, 2, sampler);
    ctx->draw(3, 1, 0, 0);
  }
  string_ref get_name() override { return stref_s("simple_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
    timer.release();
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
