#include "rendering.hpp"
#include "script.hpp"

#include "scene.hpp"

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

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
    distance = 6.0f;
    mx       = 0.0f;
    my       = 0.0f;
    look_at  = float3(0.0f, -4.0f, 0.0f);
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

Camera g_camera;

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
    aint4     material_textures;
    afloat4x4 model;
  };

  static_assert(sizeof(Push_Constants) == 80, "Uniform packing is wrong");

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
      PBR_Material & mat = model->materials[mesh_index];
      Push_Constants pc;
      MEMZERO(pc);
      pc.material_textures.x = mat.albedo_id;
      pc.material_textures.y = -1;
      pc.material_textures.z = -1;
      pc.material_textures.w = -1;
      pc.model = rotate(float4x4(1.0f), (float)0.0f, // timer.cur_time,
                        float3(0.0f, 1.0f, 0.0f)) *
                 scale(float4x4(1.0f), float3(1.0f, -1.0f, 1.0f) * 0.04f);
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
  Resource_ID        vs;
  Resource_ID        ps;
  u32                width, height;
  Resource_ID        uniform_buffer;
  Resource_ID        staging_buffer;
  Resource_ID        texture_sampler;
  bool               dummy_initialized;
  Resource_ID        dummy_texture;
  u32                current_streaming_id;
  Array<GPUMesh>     gpu_meshes;
  Array<Resource_ID> gpu_textures;
  PBR_Model          model;

  struct Uniform {
    afloat4x4 viewproj;
  };

  static_assert(sizeof(Uniform) == 64, "Uniform packing is wrong");

  public:
  Opaque_Pass() {
    vs.reset();
    ps.reset();
    width  = 0;
    height = 0;
    uniform_buffer.reset();
    staging_buffer.reset();
    texture_sampler.reset();
    dummy_initialized = false;
    dummy_texture.reset();
    current_streaming_id = 0;
    gpu_textures.init();
    gpu_meshes.init();
    model = load_gltf_pbr(stref_s("models/low_poly_ellie/scene.gltf"));
    current_streaming_id = 0;
    gpu_meshes.resize(model.meshes.size);
    ito(model.meshes.size) { gpu_meshes[i].init(&model, i); }
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
    if (staging_buffer.is_null() == false) {
      rm->release_resource(staging_buffer);
      staging_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    if (dummy_texture.is_null()) {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA8_UNORM;
      info.width      = 16;
      info.height     = 16;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                        (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
      dummy_texture = pc->create_image(info);
    }
    if (gpu_textures.size == 0) {
      gpu_textures.resize(model.images.size);
      ito(model.images.size) {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = model.images[i].format;
        info.width      = model.images[i].width;
        info.height     = model.images[i].height;
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                          (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
        gpu_textures[i] = pc->create_image(info);
      }
    }
    ito(model.meshes.size) { gpu_meshes[i].update_buffers(pc); }

    rd::Image2D_Info info = pc->get_swapchain_image_info();
    width                 = info.width;
    height                = info.height;
    g_camera.aspect       = (float)width / height;
    g_camera.update();
    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.0f;
      cl.g     = 0.0f;
      cl.b     = 0.0f;
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
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
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
    static string_ref            shader    = stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4) (name viewproj))
)
@(DECLARE_PUSH_CONSTANTS
  (add_field (type int4)     (name mesh_textures))
  (add_field (type float4x4) (name world_matrix))
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (array_size 1024)
  (dim 2D)
  (set 1)
  (binding 1)
  (name material_textures)
)
@(DECLARE_SAMPLER
  (set 1)
  (binding 0)
  (name my_sampler)
)
#ifdef VERTEX
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
  @(EXPORT_POSITION
      viewproj * world_matrix * float4(vertex_position, 1.0)
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
  float4 albedo = float4(0.0, 1.0, 1.0, 1.0);  
  if (mesh_textures.x >= 0) {
    albedo = texture(sampler2D(material_textures[nonuniformEXT(mesh_textures.x)], my_sampler), tex_coords);
  }
  @(EXPORT_COLOR 0 float4(albedo.rgb, 1.0));
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
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = sizeof(Uniform);
      uniform_buffer      = pc->create_buffer(buf_info);
    }
    if (current_streaming_id < gpu_textures.size) {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      buf_info.size       = 1 << 12;
      staging_buffer      = pc->create_buffer(buf_info);
    }
    if (texture_sampler.is_null()) {
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
      texture_sampler     = pc->create_sampler(info);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (dummy_initialized == false) {
      u32 *ptr = (u32 *)ctx->map_buffer(staging_buffer);
      ito(16 * 16) { ptr[i] = 0xff0000ffu; }
      ctx->unmap_buffer(staging_buffer);
      ctx->copy_buffer_to_image(staging_buffer, 0, dummy_texture, 0, 0);
      dummy_initialized = true;
    } else if (current_streaming_id < gpu_textures.size) {
      void *ptr = ctx->map_buffer(staging_buffer);
      memcpy(ptr, model.images[current_streaming_id].data,
             model.images[current_streaming_id].get_size_in_bytes());
      ctx->unmap_buffer(staging_buffer);
      ctx->copy_buffer_to_image(staging_buffer, 0,
                                gpu_textures[current_streaming_id], 0, 0);
      current_streaming_id++;
    }
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

      ptr->viewproj = g_camera.viewproj();
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ito(gpu_textures.size) {
      if (i < current_streaming_id) {
        ctx->bind_image(1, 1, i, gpu_textures[i], 0, 1, 0, 1);
      } else {
        ctx->bind_image(1, 1, i, dummy_texture, 0, 1, 0, 1);
      }
    }
    ctx->bind_sampler(1, 0, texture_sampler);
    ito(model.meshes.size) { gpu_meshes[i].draw(ctx); }
  }
  string_ref get_name() override { return stref_s("opaque_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    ito(model.meshes.size) { gpu_meshes[i].release(rm); }
    ito(gpu_textures.size) { rm->release_resource(gpu_textures[i]); }
    rm->release_resource(vs);
    rm->release_resource(ps);
    gpu_meshes.release();
    gpu_textures.release();
  }
};

class Postprocess_Pass : public rd::IPass {
  Resource_ID cs;
  u32         width, height;
  Resource_ID uniform_buffer;
  Resource_ID sampler;
  Resource_ID input_image;
  Resource_ID output_image;
  struct Feedback_Buffer {
    Resource_ID buffer;
    Resource_ID fence;
    bool        in_fly;
    void        reset() {
      in_fly = false;
      buffer.reset();
      fence.reset();
    }
    void release(rd::IResource_Manager *rm) {
      if (buffer.is_null() == false) rm->release_resource(buffer);
      if (fence.is_null() == false) rm->release_resource(fence);
      reset();
    }
  } feedback_buffer;
  Timer timer;

  struct Uniform {
    afloat4 color;
    ai32    control;
  } uniform_data;
  static_assert(sizeof(uniform_data) == 32, "Uniform packing is wrong");

  public:
  Postprocess_Pass() {
    timer.init();
    uniform_buffer.reset();
    sampler.reset();
    input_image.reset();
    output_image.reset();
    feedback_buffer.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4)   (name color))
  (add_field (type u32)      (name control_flags))
)
#define CONTROL_ENABLE_FEEDBACK 1
bool is_control_set(uint bits) {
  return (control_flags & bits) != 0;
}
@(DECLARE_IMAGE
  (type READ_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name my_image)
)
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 2)
  (format RGBA32_FLOAT)
  (name out_image)
)
struct BVH_Node {
  float3 min;
  float3 max;
  u32    flags;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 3)
  (type BVH_Node)
  (name bvh_nodes)
)
struct Dummy {
  uint4    flags;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 4)
  (type Dummy)
  (name dummy)
)
float4 op_laplace(int2 coords) {
  float4 val00 = image_load(my_image, coords + int2(-1, 0));
  float4 val01 = image_load(my_image, coords + int2(1, 0));
  float4 val10 = image_load(my_image, coords + int2(0, -1));
  float4 val11 = image_load(my_image, coords + int2(0, 1));
  float4 center = image_load(my_image, coords);
  float4 laplace = abs(center * 4.0 - val00 - val01 - val10 - val11) / 4.0;
  return laplace;
  // float intensity = dot(laplace, float4_splat(1.0)); 
  // return intensity > 0.5 ? float4_splat(1.0) : float4_splat(0.0);
}

@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(my_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  float2 uv = GLOBAL_THREAD_INDEX.xy / dim.xy;
  float4 in_val = op_laplace(int2(GLOBAL_THREAD_INDEX.xy));
  if (dot(in_val, float4_splat(1.0)) < 0.01) {
    in_val = float4(float2(LOCAL_THREAD_INDEX.xy) / 16.0, 0.0, 1.0);
  }
  //image_load(my_image, GLOBAL_THREAD_INDEX.xy);
  in_val = pow(in_val, float4(1.0/2.2));
  if (GLOBAL_THREAD_INDEX.y == 777 && is_control_set(CONTROL_ENABLE_FEEDBACK)) {
    Dummy d;
    d.flags.x = GLOBAL_THREAD_INDEX.x;
    d.flags.y = GLOBAL_THREAD_INDEX.y;
    d.flags.z = GLOBAL_THREAD_INDEX.z;
    d.flags.w = 666;
    buffer_store(dummy, GLOBAL_THREAD_INDEX.x, d);
  }
  image_store(out_image, GLOBAL_THREAD_INDEX.xy, in_val);
@(END)
)"),
                                 NULL, 0);
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = sizeof(uniform_data);
      uniform_buffer      = pc->create_buffer(buf_info);
    }
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
    input_image         = pc->get_resource(stref_s("opaque_pass/rt0"));
    rd::Image_Info info = pc->get_image_info(input_image);

    if (output_image.is_null() || width != info.width ||
        height != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      width  = info.width;
      height = info.height;
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA32_FLOAT;
      info.width      = width;
      info.height     = height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      output_image = pc->create_image(info);
      pc->assign_name(output_image, stref_s("postprocess/img0"));
    }
    if (feedback_buffer.in_fly == false) {
      if (feedback_buffer.buffer.is_null()) {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits      = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits    = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
        buf_info.size          = 1 << 12;
        feedback_buffer.buffer = pc->create_buffer(buf_info);
      }
      feedback_buffer.fence = pc->get_fence(rd::Fence_Position::PASS_FINISED);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (input_image.is_null()) return;
    timer.update();
    ctx->CS_set_shader(cs);
    {
      Uniform *ptr = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->color   = float4(0.5f + 0.5f * std::cos(timer.cur_time), 0.0f, 0.0f,
                          0.5f + 0.5f * std::cos(timer.cur_time));
      ptr->control = 0;
      ptr->control |= feedback_buffer.in_fly ? 0 : 1;
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_rw_image(0, 1, 0, input_image, 0, 1, 0, 1);
    ctx->bind_rw_image(0, 2, 0, output_image, 0, 1, 0, 1);
    if (feedback_buffer.in_fly == false) {
      ctx->bind_storage_buffer(0, 4, feedback_buffer.buffer, 0, 1 << 12);
      feedback_buffer.in_fly = true;
    } else {
      if (ctx->get_fence_state(feedback_buffer.fence)) {
        u32 *ptr = (u32 *)ctx->map_buffer(feedback_buffer.buffer);
        fprintf(stdout, "feedback buffer is finished: %i %i %i %i ... \n",
                ptr[0], ptr[1], ptr[2], ptr[3]);
        ctx->unmap_buffer(feedback_buffer.buffer);
        feedback_buffer.in_fly = false;
      } else {
        fprintf(stdout, "feedback buffer is buisy\n");
      }
    }
    // ctx->bind_sampler(0, 2, sampler);
    ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
  }
  string_ref get_name() override { return stref_s("postprocess_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(feedback_buffer.buffer);
    timer.release();
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
    texture(sampler2D(my_image, my_sampler), tex_coords)
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
    my_image = pc->get_resource(stref_s("postprocess/img0"));
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

class Event_Consumer : public rd::IEvent_Consumer {
  i32 last_m_x = -1;
  i32 last_m_y = -1;

  public:
  void consume(void *_event) override {

    SDL_Event *event = (SDL_Event *)_event;
    if (event->type == SDL_MOUSEMOTION) {
      SDL_MouseMotionEvent *m       = (SDL_MouseMotionEvent *)event;
      i32                   cur_m_x = m->x;
      i32                   cur_m_y = m->y;
      if ((m->state & 1) != 0 && last_m_x > 0) {
        i32 dx = cur_m_x - last_m_x;
        i32 dy = cur_m_y - last_m_y;
        g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
        g_camera.theta += (float)(dy)*5.0e-3f;
      }
      last_m_x = cur_m_x;
      last_m_y = cur_m_y;
    }
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  g_camera.init();
  Event_Consumer event_consumer;
  rd::Pass_Mng * pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  pmng->set_event_consumer(&event_consumer);
  pmng->add_pass(rd::Pass_t::RENDER, new Opaque_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Postprocess_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, new Merge_Pass);
  pmng->loop();
  return 0;
}
