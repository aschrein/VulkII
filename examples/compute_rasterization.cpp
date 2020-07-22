#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "script.hpp"

#include "scene.hpp"

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

#include <imgui.h>
#include <imgui/examples/imgui_impl_sdl.h>

Config g_config;
Camera g_camera;
Scene  g_scene;

static void init_traverse(List *l) {
  if (l == NULL) return;
  if (l->child) {
    init_traverse(l->child);
    init_traverse(l->next);
  } else {
    if (l->cmp_symbol("camera")) {
      g_camera.traverse(l->next);
    } else if (l->cmp_symbol("config")) {
      g_config.traverse(l->next);
    }
  }
}

static int g_init = []() {
  TMP_STORAGE_SCOPE;
  g_camera.init();
  g_config.init(stref_s(R"(
(config
 (add bool enable_rasterization_pass 1)
 (add bool enable_compute_depth 1)
 (add bool enable_compute_render_pass 1)
 (add bool enable_meshlets_render_pass 1)
 (add u32 g_buffer_width 512 (min 4) (max 1024))
 (add u32 g_buffer_height 512 (min 4) (max 1024))
)
)"));

  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
    init_traverse(cur);
  }
  g_scene.init();
  g_scene.load_mesh(stref_s("HIGH"), stref_s("models/light/scene.gltf"));
  g_scene.load_mesh(stref_s("LOW"), stref_s("models/light/scene_low.gltf"));
  Node *high = g_scene.get_node(stref_s("HIGH"));
  Node *low  = g_scene.get_node(stref_s("LOW"));
  ito(10) {
    jto(10) {
      high->clone()->translate(float3((f32)i * 2.0f, 0.0f, (f32)j * 8.0f));
      low->clone()->translate(float3((f32)i * 2.0f, 0.0f, (f32)j * 8.0f));
    }
  }
  return 0;
}();

static_defer({
  FILE *scene_dump = fopen("scene_state", "wb");
  fprintf(scene_dump, "(\n");
  defer(fclose(scene_dump));
  g_camera.dump(scene_dump);
  g_config.dump(scene_dump);
  fprintf(scene_dump, ")\n");
});

class Depth_PrePass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  Resource_ID uniform_buffer;
  struct Uniform {
    afloat4x4 viewproj;
  };
  static_assert(sizeof(Uniform) == 64, "Uniform packing is wrong");

  public:
  Depth_PrePass() {
    vs.reset();
    ps.reset();
    uniform_buffer.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
    g_scene.on_pass_end(rm);
  }
  void on_begin(rd::IResource_Manager *pc) override {
    g_camera.aspect = (float)g_config.get_u32("g_buffer_width") /
                      g_config.get_u32("g_buffer_height");
    g_camera.update();
    {
      rd::Clear_Depth cl;
      cl.clear = true;
      cl.d     = 0.0f;
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::D32_FLOAT;
      info.width      = g_config.get_u32("g_buffer_width");
      info.height     = g_config.get_u32("g_buffer_height");
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      pc->add_depth_target(stref_s("depth_prepas/ds"), info, 0, 0, cl);
    }
    static string_ref            shader    = stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4) (name viewproj))
)

@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name world_transform))
)

struct Instance_Info {
  i32 albedo_id;
  i32 arm_id;
  i32 normal_id;
  i32 pad;
  float4x4 model;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type Instance_Info)
  (name instance_infos)
)
#ifdef VERTEX
@(DECLARE_INPUT (location 0) (type float3) (name POSITION))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))

@(DECLARE_OUTPUT (location 1) (type uint) (name PIXEL_INSTANCE_ID))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_INSTANCE_ID = INSTANCE_INDEX;
  float3 position = POSITION;
  // float4x4 world_matrix = buffer_load(instance_infos, INSTANCE_INDEX).model;
  @(EXPORT_POSITION
      viewproj * world_transform * float4(position, 1.0)
  );
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))

@(DECLARE_INPUT (location 1) (type "flat uint") (name PIXEL_INSTANCE_ID))

@(ENTRY)
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
    g_scene.on_pass_begin(pc);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.get_bool("enable_rasterization_pass") == false) return;
    setup_default_state(ctx);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)g_config.get_u32("g_buffer_width"),
                      (float)g_config.get_u32("g_buffer_height"), 0.0f, 1.0f);
    ctx->set_scissor(0, 0, g_config.get_u32("g_buffer_width"),
                     g_config.get_u32("g_buffer_height"));
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    rs_state.line_width   = 1.0f;
    rs_state.depth_bias   = 0.0f;
    ctx->RS_set_state(rs_state);
    {
      Uniform *ptr  = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->viewproj = g_camera.viewproj();
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    g_scene.gfx_exec_low(ctx);
  }
  string_ref get_name() override { return stref_s("depth_prepass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
  }
};

class Opaque_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  Resource_ID uniform_buffer;
  Resource_ID texture_sampler;
  struct Uniform {
    afloat4x4 viewproj;
  };
  static_assert(sizeof(Uniform) == 64, "Uniform packing is wrong");

  public:
  Opaque_Pass() {
    vs.reset();
    ps.reset();
    uniform_buffer.reset();
    texture_sampler.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
    g_scene.on_pass_end(rm);
  }
  void on_begin(rd::IResource_Manager *pc) override {
    // rd::Image2D_Info info = pc->get_swapchain_image_info();
    //    width                 = info.width;
    // height                = info.height;
    g_camera.aspect = (float)g_config.get_u32("g_buffer_width") /
                      g_config.get_u32("g_buffer_height");
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
      rt0_info.width      = g_config.get_u32("g_buffer_width");
      rt0_info.height     = g_config.get_u32("g_buffer_height");
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
      info.width      = g_config.get_u32("g_buffer_width");
      info.height     = g_config.get_u32("g_buffer_height");
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
  (add_field (type float4x4)  (name world_transform))
)

struct Instance_Info {
  i32 albedo_id;
  i32 arm_id;
  i32 normal_id;
  i32 pad;
  float4x4 model;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type Instance_Info)
  (name instance_infos)
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
@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))

//@(DECLARE_INPUT (location 2) (type float3) (name BINORMAL))
//@(DECLARE_INPUT (location 3) (type float3) (name TANGENT))
//@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))
//@(DECLARE_INPUT (location 5) (type float2) (name TEXCOORD1))
//@(DECLARE_INPUT (location 6) (type float2) (name TEXCOORD2))
//@(DECLARE_INPUT (location 7) (type float2) (name TEXCOORD3))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))

//@(DECLARE_OUTPUT (location 2) (type float3) (name PIXEL_BINORMAL))
//@(DECLARE_OUTPUT (location 3) (type float3) (name PIXEL_TANGENT))
//@(DECLARE_OUTPUT (location 4) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_OUTPUT (location 5) (type uint) (name PIXEL_INSTANCE_ID))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_NORMAL    = NORMAL;
  //PIXEL_BINORMAL  = BINORMAL;
  //PIXEL_TANGENT   = TANGENT;
  //PIXEL_TEXCOORD0 = TEXCOORD0;
  PIXEL_INSTANCE_ID = INSTANCE_INDEX;
  float3 position = POSITION;
  // float4x4 world_matrix = buffer_load(instance_infos, INSTANCE_INDEX).model;
  @(EXPORT_POSITION
      viewproj * world_transform * float4(position, 1.0)
  );
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
//@(DECLARE_INPUT (location 2) (type float3) (name PIXEL_BINORMAL))
//@(DECLARE_INPUT (location 3) (type float3) (name PIXEL_TANGENT))
//@(DECLARE_INPUT (location 4) (type float3) (name PIXEL_TEXCOORD0))
@(DECLARE_INPUT (location 5) (type "flat uint") (name PIXEL_INSTANCE_ID))

@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  float4 albedo = float4(0.0, 1.0, 1.0, 1.0);
  float4 color = float4_splat(1.0) * (0.5 + 0.5 * dot(PIXEL_NORMAL.rgb, normalize(float3(1.0, 1.0, 1.0))));
  /*i32 albedo_id = buffer_load(instance_infos, instance_index).albedo_id;
  if (albedo_id >= 0) {
    albedo = texture(sampler2D(material_textures[nonuniformEXT(albedo_id)], my_sampler), tex_coords);
  }*/
  @(EXPORT_COLOR 0 color);
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
    g_scene.on_pass_begin(pc);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.get_bool("enable_rasterization_pass") == false) return;
    setup_default_state(ctx);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)g_config.get_u32("g_buffer_width"),
                      (float)g_config.get_u32("g_buffer_height"), 0.0f, 1.0f);
    ctx->set_scissor(0, 0, g_config.get_u32("g_buffer_width"),
                     g_config.get_u32("g_buffer_height"));
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    rs_state.line_width   = 1.0f;
    rs_state.depth_bias   = 0.0f;
    ctx->RS_set_state(rs_state);
    {
      Uniform *ptr  = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->viewproj = g_camera.viewproj();
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_sampler(1, 0, texture_sampler);
    g_scene.gfx_exec(ctx);
  }
  string_ref get_name() override { return stref_s("opaque_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
  }
};

class Compute_Render_Pass : public rd::IPass {
  Resource_ID cs;
  Resource_ID clear_cs;
  Resource_ID output_image;
  Resource_ID output_depth;
  Resource_ID uniform_buffer;

  public:
  Compute_Render_Pass() {
    cs.reset();
    clear_cs.reset();
    output_image.reset();
    output_depth.reset();
    uniform_buffer.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = 68;
      uniform_buffer      = pc->create_buffer(buf_info);
    }
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type u32)       (name control_flags))
)

@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name world_transform))
  (add_field (type u32)       (name index_count))
  (add_field (type u32)       (name first_index))
  (add_field (type i32)       (name vertex_offset))
)

#define CONTROL_DEPTH_ENABLE 1
#define is_control(flag) (control_flags & flag) != 0

@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 2)
  (format R32_UINT)
  (name out_depth)
)
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 0)
  (type u32)
  (name index_buffer)
)

struct Vertex {
  float3 position;
  float3 normal;
};

#define FETCH_FLOAT3

#ifdef FETCH_FLOAT3
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float3)
  (name position_buffer)
)
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float3)
  (name normal_buffer)
)
Vertex fetch(u32 index) {
  Vertex o;  
  o.position = buffer_load(position_buffer, index);
  o.normal = buffer_load(normal_buffer, index);
  return o;
}
#else
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float)
  (name position_buffer)
)
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float)
  (name normal_buffer)
)

Vertex fetch(u32 index) {
  Vertex o;  
  o.position.x = buffer_load(position_buffer, index * 3 + 0);
  o.position.y = buffer_load(position_buffer, index * 3 + 1);
  o.position.z = buffer_load(position_buffer, index * 3 + 2);
  o.normal.x = buffer_load(normal_buffer, index * 3 + 0);
  o.normal.y = buffer_load(normal_buffer, index * 3 + 1);
  o.normal.z = buffer_load(normal_buffer, index * 3 + 2);
  return o;
}
#endif

@(GROUP_SIZE 256 1 1)
@(ENTRY)
  
  u32 triangle_index =  GLOBAL_THREAD_INDEX.x;
  if (triangle_index > index_count / 3)
    return;

  u32 i0 = buffer_load(index_buffer, first_index + triangle_index * 3 + 0);
  u32 i1 = buffer_load(index_buffer, first_index + triangle_index * 3 + 1);
  u32 i2 = buffer_load(index_buffer, first_index + triangle_index * 3 + 2);
   
  Vertex v0 = fetch(vertex_offset + i0);
  Vertex v1 = fetch(vertex_offset + i1);
  Vertex v2 = fetch(vertex_offset + i2);

  float4 pp0 = mul4(viewproj * world_transform, float4(v0.position, 1.0));
  float4 pp1 = mul4(viewproj * world_transform, float4(v1.position, 1.0));
  float4 pp2 = mul4(viewproj * world_transform, float4(v2.position, 1.0));
  pp0.xyz /= pp0.w;
  pp1.xyz /= pp1.w;
  pp2.xyz /= pp2.w;
  {
    float2 e0 = pp0.xy - pp2.xy; 
    float2 e1 = pp1.xy - pp0.xy; 
    float2 e2 = pp2.xy - pp1.xy; 
    float2 n0 = float2(e0.y, -e0.x);
    if (dot(e1, n0) > 0.0)
      return;
  }
  float area = 1.0;
  float b0 = 1.0 / 3.0;
  float b1 = 1.0 / 3.0;
  float b2 = 1.0 / 3.0;
  b0 /= area;
  b1 /= area;
  b2 /= area;
  float z = 1.0 / (b0 / pp0.w + b1 / pp1.w + b2 / pp2.w);

  float3 pp = pp0.xyz * b0 + pp1.xyz * b1 + pp2.xyz * b2;
  float3 n = normalize(z * (v0.normal * b0 + v1.normal * b1 + v1.normal * b2));
  int2 dim = imageSize(out_image);
  if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
    return;
  i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
  i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
  if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
    float4 color = float4_splat(1.0) * (0.5 + 0.5 * dot(n, normalize(float3(1.0, 1.0, 1.0))));
    if (is_control(CONTROL_DEPTH_ENABLE)) {
      u32 depth = u32(1.0 / pp.z);
      if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
        image_store(out_image, int2(x, y), color);
      }
    } else {
      image_store(out_image, int2(x, y), color);
    }
  }
  
@(END)
)"),
                                 NULL, 0);
    if (clear_cs.is_null())
      clear_cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 2)
  (format R32_UINT)
  (name out_depth)
)
@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  image_store(out_image, int2(GLOBAL_THREAD_INDEX.xy), float4(0.0, 0.0, 0.0, 1.0));
  image_store(out_depth, int2(GLOBAL_THREAD_INDEX.xy), uint4(1 << 31, 0, 0, 0));
@(END)
)"),
                                       NULL, 0);
    rd::Image_Info info;
    MEMZERO(info);
    if (output_image.is_null() == false)
      info = pc->get_image_info(output_image);
    if (output_image.is_null() ||
        g_config.get_u32("g_buffer_width") != info.width ||
        g_config.get_u32("g_buffer_height") != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      if (output_depth.is_null() == false) pc->release_resource(output_depth);
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::RGBA32_FLOAT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_image = pc->create_image(info);
      }
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::R32_UINT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_depth = pc->create_image(info);
      }

      pc->assign_name(output_image, stref_s("compute_render/img0"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.get_bool("enable_compute_render_pass") == false) return;
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->CS_set_shader(clear_cs);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth,
                       (u32)rd::Access_Bits::SHADER_WRITE |
                           (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 1, 0, output_image, rd::Image_Range::top_level(),
                         rd::Format::NATIVE);
    ctx->bind_rw_image(0, 2, 0, output_depth, rd::Image_Range::top_level(),
                         rd::Format::NATIVE);
    ctx->dispatch((g_config.get_u32("g_buffer_width") + 15) / 16,
                  (g_config.get_u32("g_buffer_height") + 15) / 16, 1);
    ctx->CS_set_shader(cs);
    u32 control_flags = 0;
    control_flags |= (g_config.get_bool("enable_compute_depth") ? 1 : 0) << 0;

    g_config.get_u32("triangles_per_lane") =
        MAX(0, MIN(128, g_config.get_u32("triangles_per_lane")));
    {
      struct Uniform {
        float4x4 viewproj;
        u32      control;
      };
      Uniform *ptr  = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->viewproj = g_camera.viewproj();
      ptr->control  = control_flags;
      ctx->unmap_buffer(uniform_buffer);
      ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    }
    g_scene.gfx_dispatch(ctx, 0);
  }
  string_ref get_name() override { return stref_s("compute_render_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(output_image);
  }
};

class Meshlet_Compute_Render_Pass : public rd::IPass {
  Resource_ID cs;
  Resource_ID clear_cs;
  Resource_ID output_image;
  Resource_ID output_depth;
  Resource_ID uniform_buffer;

  public:
  Meshlet_Compute_Render_Pass() {
    cs.reset();
    clear_cs.reset();
    output_image.reset();
    output_depth.reset();
    uniform_buffer.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = 68;
      uniform_buffer      = pc->create_buffer(buf_info);
    }
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(

@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type u32)       (name control_flags))
)

@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name world_transform))
  (add_field (type u32)       (name num_meshlets))
  (add_field (type u32)       (name meshlet_offset))
  (add_field (type u32)       (name index_offset))
  (add_field (type u32)       (name vertex_offset))
)

#define CONTROL_DEPTH_ENABLE 1
#define is_control(flag) (control_flags & flag) != 0

@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 2)
  (format R32_UINT)
  (name out_depth)
)
struct Meshlet {
  u32 vertex_offset;
  u32 index_offset;
  u32  triangle_count;
  u32  vertex_count;
  float4 sphere;
  float4 cone_apex;
  float4 cone_axis_cutoff;
  u32 cone_pack;
};

struct Vertex {
  float3 position;
  float3 normal;
};

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 1)
  (binding 0)
  (type Meshlet)
  (name meshlet_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 0)
  (type u32)
  (name index_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float3)
  (name position_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float3)
  (name normal_buffer)
)

Vertex fetch(u32 index) {
  Vertex o;  
  o.position = buffer_load(position_buffer, index);
  o.normal = buffer_load(normal_buffer, index);
  return o;
}

u32 fetch_index(u32 index) {
  u32 raw = buffer_load(index_buffer, (index_offset + index) / 4);
  u32 sub_index = index & 0x3;
  return (raw >> (sub_index * 8)) & 0xffu;
}

//shared u32    gs_indices[64];
//shared float3 gs_vertices[64];
//shared Meshlet meshlet;

@(GROUP_SIZE 256 1 1)
@(ENTRY)
  
  u32 meshlet_index = GROUPT_INDEX.x;
  if (meshlet_index > num_meshlets)
    return;

  //if (LOCAL_THREAD_INDEX.x == 0)
  Meshlet  meshlet = buffer_load(meshlet_buffer, meshlet_offset + meshlet_index);
  //barrier();
  {
    float4 cp = mul4(viewproj, float4(meshlet.cone_apex.xyz, 1.0));
    float4 ca = mul4(viewproj, float4(meshlet.cone_apex.xyz + meshlet.cone_axis_cutoff.xyz, 1.0));
    if (ca.z > cp.z)
      return;
  }
  /*if (LOCAL_THREAD_INDEX.x < meshlet.vertex_count)
    gs_vertices[LOCAL_THREAD_INDEX.x] = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + LOCAL_THREAD_INDEX.x);

  barrier();*/

  u32 triangle_index = LOCAL_THREAD_INDEX.x;

  if (triangle_index > meshlet.triangle_count)
    return;

  u32 i0 = fetch_index(meshlet.index_offset + triangle_index * 3 + 0);
  u32 i1 = fetch_index(meshlet.index_offset + triangle_index * 3 + 1);
  u32 i2 = fetch_index(meshlet.index_offset + triangle_index * 3 + 2);
   
  // Vertex v0 = gs_vertices[i0];//fetch(vertex_offset + meshlet.vertex_offset + i0);
  // Vertex v0 = fetch(vertex_offset + meshlet.vertex_offset + i0);
  // Vertex v1 = gs_vertices[i1];//fetch(vertex_offset + meshlet.vertex_offset + i1);
  // Vertex v1 = fetch(vertex_offset + meshlet.vertex_offset + i1);
  // Vertex v2 = gs_vertices[i2];//fetch(vertex_offset + meshlet.vertex_offset + i2);
  // Vertex v2 = fetch(vertex_offset + meshlet.vertex_offset + i2);
  
  //float3 p0 = gs_vertices[i0];
  //float3 p1 = gs_vertices[i1];
  //float3 p2 = gs_vertices[i2];

  float3 p0 = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + i0);
  float3 p1 = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + i1);
  float3 p2 = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + i2);

  float4 pp0 = mul4(viewproj * world_transform, float4(p0, 1.0));
  float4 pp1 = mul4(viewproj * world_transform, float4(p1, 1.0));
  float4 pp2 = mul4(viewproj * world_transform, float4(p2, 1.0));
  pp0.xyz /= pp0.w;
  pp1.xyz /= pp1.w;
  pp2.xyz /= pp2.w;
  {
    float2 e0 = pp0.xy - pp2.xy; 
    float2 e1 = pp1.xy - pp0.xy; 
    float2 e2 = pp2.xy - pp1.xy; 
    float2 n0 = float2(e0.y, -e0.x);
    if (dot(e1, n0) > 0.0)
      return;
  }
  float area = 1.0;
  float b0 = 1.0 / 3.0;
  float b1 = 1.0 / 3.0;
  float b2 = 1.0 / 3.0;
  b0 /= area;
  b1 /= area;
  b2 /= area;
  float z = 1.0 / (b0 / pp0.w + b1 / pp1.w + b2 / pp2.w);

  float3 pp = pp0.xyz * b0 + pp1.xyz * b1 + pp2.xyz * b2;
  
  int2 dim = imageSize(out_image);
  if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
    return;
  i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
  i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
  if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
    float3 n0 = buffer_load(normal_buffer, vertex_offset + meshlet.vertex_offset + i0);
    float3 n1 = buffer_load(normal_buffer, vertex_offset + meshlet.vertex_offset + i1);
    float3 n2 = buffer_load(normal_buffer, vertex_offset + meshlet.vertex_offset + i2);
    float3 n = normalize(z * (n0 * b0 + n1 * b1 + n1 * b2));
    float4 color = float4(
        float(((31 * meshlet.vertex_offset) >> 0) % 255) / 255.0,
        float(((31 * meshlet.vertex_offset) >> 8) % 255) / 255.0,
        float(((31 * meshlet.vertex_offset) >> 16) % 255) / 255.0,
        1.0
        );
    //color = float4((0.5 * n + float3_splat(0.5)), 1.0);
    color = float4_splat(1.0) * (0.5 + 0.5 * dot(n, normalize(float3(1.0, 1.0, 1.0))));
    u32 depth = u32(1.0 / pp.z);
    if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
      image_store(out_image, int2(x, y), color);
    }
  }
@(END)
)"),
                                 NULL, 0);
    if (clear_cs.is_null())
      clear_cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 2)
  (format R32_UINT)
  (name out_depth)
)
@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  image_store(out_image, int2(GLOBAL_THREAD_INDEX.xy), float4(0.0, 0.0, 0.0, 1.0));
  image_store(out_depth, int2(GLOBAL_THREAD_INDEX.xy), uint4(1 << 31, 0, 0, 0));
@(END)
)"),
                                       NULL, 0);
    rd::Image_Info info;
    MEMZERO(info);
    if (output_image.is_null() == false)
      info = pc->get_image_info(output_image);
    if (output_image.is_null() ||
        g_config.get_u32("g_buffer_width") != info.width ||
        g_config.get_u32("g_buffer_height") != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      if (output_depth.is_null() == false) pc->release_resource(output_depth);
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::RGBA32_FLOAT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_image = pc->create_image(info);
      }
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::R32_UINT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_depth = pc->create_image(info);
      }
      pc->assign_name(output_image, stref_s("meshlet_render/img0"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.get_bool("enable_meshlets_render_pass") == false) return;
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->CS_set_shader(clear_cs);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth,
                       (u32)rd::Access_Bits::SHADER_WRITE |
                           (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 1, 0, output_image, rd::Image_Range::top_level(),
                         rd::Format::NATIVE);
    ctx->bind_rw_image(0, 2, 0, output_depth, rd::Image_Range::top_level(),
                         rd::Format::NATIVE);
    ctx->dispatch((g_config.get_u32("g_buffer_width") + 15) / 16,
                  (g_config.get_u32("g_buffer_height") + 15) / 16, 1);
    ctx->CS_set_shader(cs);
    float4x4 vp = g_camera.viewproj();
    {
      struct Uniform {
        float4x4 viewproj;
        u32      control;
      };
      Uniform *ptr  = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->viewproj = g_camera.viewproj();
      ptr->control  = 0;
      ctx->unmap_buffer(uniform_buffer);
      ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    }
    g_scene.gfx_dispatch_meshlets(ctx, 0);
  }
  string_ref get_name() override { return stref_s("meshlet_render_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(output_image);
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
    cs.reset();
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
  
  float4 in_val = image_load(my_image, GLOBAL_THREAD_INDEX.xy);
  in_val = pow(in_val, float4(1.0/2.2));
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
    ctx->image_barrier(input_image, (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 1, 0, input_image, rd::Image_Range::top_level(),
                         rd::Format::NATIVE);
    ctx->bind_rw_image(0, 2, 0, output_image, rd::Image_Range::top_level(),
                         rd::Format::NATIVE);
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

class GUI_Pass : public IGUI_Pass {
  public:
  void on_gui(rd::IResource_Manager *pc) override {
    {
      ImGui::Begin("Config");
      g_config.on_imgui();
      ImGui::LabelText("hw rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("opaque_pass")));
      ImGui::LabelText("sw rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("compute_render_pass")));
      ImGui::LabelText("meshlet rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("meshlet_render_pass")));
      ImGui::LabelText("depth prepass", "%f ms",
                       pc->get_pass_duration(stref_s("depth_prepass")));
      ImGui::End();
    }
    {
      ImGui::Begin("sw rasterization");
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("compute_render/img0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }
      ImGui::End();
      ImGui::Begin("hw rasterization");
      auto  wpos        = ImGui::GetCursorScreenPos();
      auto  wsize       = ImGui::GetWindowSize();
      float height_diff = 24;
      if (wsize.y < height_diff + 2) {
        wsize.y = 2;
      } else {
        wsize.y = wsize.y - height_diff;
      }
      ImGuiIO &io = ImGui::GetIO();
      // g_config.get_u32("g_buffer_width")  = wsize.x;
      // g_config.get_u32("g_buffer_height") = wsize.y;
      if (ImGui::IsWindowHovered()) {
        auto scroll_y = ImGui::GetIO().MouseWheel;
        if (scroll_y) {
          g_camera.distance += g_camera.distance * 2.e-1 * scroll_y;
          g_camera.distance = clamp(g_camera.distance, 1.0e-3f, 1000.0f);
        }
        f32 camera_speed = 2.0f;
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_LSHIFT]) {
          camera_speed = 20.0f;
        }
        float3 camera_diff = float3(0.0f, 0.0f, 0.0f);
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_W]) {
          camera_diff += g_camera.look;
        }
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_S]) {
          camera_diff -= g_camera.look;
        }
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_A]) {
          camera_diff -= g_camera.right;
        }
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_D]) {
          camera_diff += g_camera.right;
        }
        if (dot(camera_diff, camera_diff) > 1.0e-3f) {
          g_camera.look_at +=
              glm::normalize(camera_diff) * camera_speed * (float)timer.dt;
        }
        ImVec2 mpos    = ImGui::GetMousePos();
        i32    cur_m_x = mpos.x;
        i32    cur_m_y = mpos.y;
        if (io.MouseDown[0] && last_m_x > 0) {
          i32 dx = cur_m_x - last_m_x;
          i32 dy = cur_m_y - last_m_y;
          g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
          g_camera.theta += (float)(dy)*5.0e-3f;
        }
        last_m_x = cur_m_x;
        last_m_y = cur_m_y;
      }
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("opaque_pass/rt0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
      ImGui::Begin("meshlet rasterization");
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("meshlet_render/img0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
      ImGui::Begin("depth prepass");
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("depth_prepas/ds"));
        ImGui::Image((ImTextureID)(intptr_t)(img.data),
                     ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
    }
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  GUI_Pass *    gui  = new GUI_Pass;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  pmng->set_event_consumer(gui);
  pmng->add_pass(rd::Pass_t::RENDER, new Depth_PrePass);
  pmng->add_pass(rd::Pass_t::RENDER, new Opaque_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Compute_Render_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Meshlet_Compute_Render_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Postprocess_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, gui);
  pmng->loop();
  return 0;
}
