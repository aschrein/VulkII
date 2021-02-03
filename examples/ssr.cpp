#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC

#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "script.hpp"

#include <3rdparty/half.hpp>
#include <atomic>
#include <condition_variable>
#include <imgui.h>
#include <mutex>
#include <thread>

class GBufferPass : public IPass {
  ~GBufferPass() = default;

  public:
  static constexpr char const *NAME = "GBuffer Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(normal_rt);                                                                             \
  RESOURCE(depth_rt);                                                                              \
  RESOURCE(gbuffer_vs);                                                                            \
  RESOURCE(gbuffer_ps);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  void release() override {
    timestamps.release(rctx->factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) rctx->factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
    delete this;
  }
#undef RESOURCE_LIST

  u32 width  = 0;
  u32 height = 0;
  // BufferThing bthing{};

  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};
  RenderingContext *          rctx = NULL;

  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  static GBufferPass *create(RenderingContext *rctx) {
    GBufferPass *o = new GBufferPass;
    o->init(rctx);
    return o;
  }
  void init(RenderingContext *rctx) {
    this->rctx = rctx;
    auto dev   = rctx->factory;
    timestamps.init(dev);
    // bthing.init(dev);
    gbuffer_vs = dev->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 view_to_proj;
  float4x4 obj_to_view;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] float3 normal  : NORMAL;
  [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.normal = mul(pc.obj_to_view, float4(input.normal.xyz, 0.0f)).xyz;
  output.uv     = input.uv;
  output.pos    = mul(pc.view_to_proj, mul(pc.obj_to_view, float4(input.pos, 1.0f)));
  return output;
}
)"),
                                    NULL, 0);
    gbuffer_ps = dev->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
};

float4 main(in PSInput input) : SV_TARGET0 {
  return float4(input.normal.xyz, 1.0f);
}
)"),
                                    NULL, 0);
    signature  = [=] {
      rd::Binding_Space_Create_Info set_info{};
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      rd::RT_Ref                  rt0{};
      rt0.format            = rd::Format::RGBA32_FLOAT;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      info.depth_target.enabled           = true;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_render_pass(info);
    }();

    pso = [=] {
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state{};
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      gfx_state.RS_set_state(rs_state);
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      gfx_state.DS_set_state(ds_state);
      rd::Blend_State bs{};
      bs.enabled = false;
      bs.color_write_mask =
          (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
          (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
      gfx_state.OM_set_blend_state(0, bs);
      gfx_state.VS_set_shader(gbuffer_vs);
      gfx_state.PS_set_shader(gbuffer_ps);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 1;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 1;
        info.offset   = 0;
        info.type     = rd::Attriute_t::NORMAL;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 2;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 2;
        info.offset   = 0;
        info.type     = rd::Attriute_t::TEXCOORD0;
        gfx_state.IA_set_attribute(info);
      }
      gfx_state.IA_set_vertex_binding(0, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();
  }
  void update_frame_buffer(RenderingContext *rctx) {
    auto dev = rctx->factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (normal_rt.is_valid()) dev->release_resource(normal_rt);
    if (depth_rt.is_valid()) dev->release_resource(depth_rt);

    normal_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
    depth_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format = rd::Format::D32_OR_R32_FLOAT;
      rt0_info.width  = width;
      rt0_info.height = height;
      rt0_info.depth  = 1;
      rt0_info.layers = 1;
      rt0_info.levels = 1;
      rt0_info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_DT | (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      return dev->create_image(rt0_info);
    }();
    frame_buffer = [=] {
      rd::Frame_Buffer_Create_Info info{};
      rd::RT_View                  rt0{};
      rt0.image  = normal_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);

      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render() {
    auto dev = rctx->factory;
    timestamps.update(dev);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f,
    // 0.0f));
    // bthing.test_buffers(dev);
    u32 width  = rctx->config->get_u32("g_buffer_width");
    u32 height = rctx->config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    struct PushConstants {
      float4x4 view_to_proj;
      float4x4 obj_to_view;
    } pc;

    float4x4 view_to_proj  = rctx->gizmo_layer->get_camera().proj;
    float4x4 world_to_view = rctx->gizmo_layer->get_camera().view;

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "GBuffer Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      pc.view_to_proj = view_to_proj;

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      ctx->bind_table(table);
      ctx->bind_graphics_pso(pso);

      if (rctx->config->get_bool("ras.render_meshlets")) {
        rctx->scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            if (auto *sc = mn->getComponent<GfxMeshletSufraceComponent>()) {
              ito(mn->getNumSurfaces()) {
                GfxMeshletSurface *gfx_meshlets = sc->get_meshlets(i);
                gfx_meshlets->iterate([](Meshlet const &meshlet) {

                });
              }
            }
          }
        });
      }
      if (rctx->config->get_bool("ras.render_geometry")) {
        rctx->scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            GfxSufraceComponent *gs           = mn->getComponent<GfxSufraceComponent>();
            float4x4             obj_to_world = mn->get_transform();
            pc.obj_to_view                    = world_to_view * obj_to_world;
            table->push_constants(&pc, 0, sizeof(pc));
            ito(gs->getNumSurfaces()) {
              GfxSurface *s = gs->getSurface(i);
              s->draw(ctx, gfx_state);
            }
          }
        });
      }
      ctx->end_render_pass();
      timestamps.end_range(ctx);
    }

    Resource_ID e = dev->end_render_pass(ctx);
    timestamps.commit(e);
  }
  GBuffer get_gbuffer() {
    GBuffer out{};
    out.normal = normal_rt;
    out.depth  = depth_rt;
    return out;
  }

  char const *getName() override { return NAME; }
  u32         getNumBuffers() override { return 2; }
  char const *getBufferName(u32 i) override {
    if (i == 0) return "GBuffer.Normal";
    return "GBuffer.Depth";
  }
  Resource_ID getBuffer(u32 i) override {
    if (i == 0) return normal_rt;
    return depth_rt;
  }
  double getLastDurationInMs() override { return timestamps.duration; }
};

class GizmoPass : public IPass {
  ~GizmoPass() = default;

  public:
  static constexpr char const *NAME = "Gizmo Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(rt);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  void release() override {
    timestamps.release(rctx->factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) rctx->factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
    delete this;
  }
#undef RESOURCE_LIST

  RenderingContext *rctx = NULL;

  u32         width  = 0;
  u32         height = 0;
  Resource_ID last_depth_rt{};

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  static GizmoPass *create(RenderingContext *rctx) {
    GizmoPass *o = new GizmoPass;
    o->init(rctx);
    return o;
  }
  void init(RenderingContext *rctx) {
    this->rctx = rctx;
    auto dev   = rctx->factory;
    timestamps.init(dev);
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      rd::RT_Ref                  rt0{};
      rt0.format            = rd::Format::RGBA32_FLOAT;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      info.depth_target.enabled           = true;
      info.depth_target.clear_depth.clear = false;
      info.depth_target.format            = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_render_pass(info);
    }();
  }
  void update_frame_buffer(RenderingContext *rctx, Resource_ID depth_rt) {
    auto dev = rctx->factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (rt.is_valid()) dev->release_resource(rt);

    rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();

    frame_buffer = [=] {
      rd::Frame_Buffer_Create_Info info{};
      rd::RT_View                  rt0{};
      rt0.image  = rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);

      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render() {
    auto dev = rctx->factory;
    timestamps.update(dev);
    u32         width    = rctx->config->get_u32("g_buffer_width");
    u32         height   = rctx->config->get_u32("g_buffer_height");
    Resource_ID depth_rt = rctx->pass_mng->getPass(GBufferPass::NAME)->getBuffer("GBuffer.Depth");
    if (this->width != width || this->height != height || last_depth_rt.data != depth_rt.data) {
      this->width   = width;
      this->height  = height;
      last_depth_rt = depth_rt;
      update_frame_buffer(rctx, depth_rt);
    }
    if (rctx->config->get_bool("gizmo.enable")) {
      auto g_camera = rctx->gizmo_layer->get_camera();
      {
        float dx = 1.0e-1f * g_camera.distance;
        rctx->gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
        rctx->gizmo_layer->draw_cylinder(g_camera.look_at,
                                         g_camera.look_at + float3{dx, 0.0f, 0.0f}, dx * 0.04f,
                                         float3{1.0f, 0.0f, 0.0f});
        rctx->gizmo_layer->draw_cylinder(g_camera.look_at,
                                         g_camera.look_at + float3{0.0f, dx, 0.0f}, dx * 0.04f,
                                         float3{0.0f, 1.0f, 0.0f});
        rctx->gizmo_layer->draw_cylinder(g_camera.look_at,
                                         g_camera.look_at + float3{0.0f, 0.0f, dx}, dx * 0.04f,
                                         float3{0.0f, 0.0f, 1.0f});
      }

      if (rctx->config->get_bool("gizmo.render_bounds")) {
        rctx->scene->traverse([&](Node *node) {
          AABB     aabb = node->getAABB();
          float4x4 t(1.0f);
          rctx->gizmo_layer->render_linebox(transform(t, aabb.min), transform(t, aabb.max),
                                            float3(1.0f, 0.0f, 0.0f));
        });
      }
      if (rctx->config->get_bool("gizmo.render_bvh")) {
        rctx->scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {

            if (auto *sc = mn->getComponent<BVHSufraceComponent>()) {
              if (sc->getBVH()) {
                render_bvh(float4x4(1.0f), sc->getBVH(), rctx->gizmo_layer);
              }
            }
          }
        });
      }
    }

    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = rctx->gizmo_layer->get_camera().viewproj();

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "Gizmo Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      rctx->gizmo_layer->render(ctx, width, height);
      ctx->end_render_pass();
      timestamps.end_range(ctx);
    }
    rctx->gizmo_layer->reset();
    Resource_ID e = dev->end_render_pass(ctx);
    timestamps.commit(e);
  }

  char const *getName() override { return NAME; }
  u32         getNumBuffers() override { return 1; }
  char const *getBufferName(u32 i) override { return "Gizmo.RT"; }
  Resource_ID getBuffer(u32 i) override { return rt; }
  double      getLastDurationInMs() override { return timestamps.duration; }
};

class HiZPass : public IPass {
  ~HiZPass() = default;

  public:
  static constexpr char const *NAME = "HiZ Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(hiz);                                                                                   \
  RESOURCE(cnt_image);                                                                             \
  RESOURCE(reset_cnt_pso);                                                                         \
  RESOURCE(pso);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE
  void release() override {
    timestamps.release(rctx->factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) rctx->factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
    delete this;
  }
#undef RESOURCE_LIST
  RenderingContext *rctx = NULL;

  static constexpr u32 SIZE_POT = 11;
  static constexpr u32 SIZE     = 1 << SIZE_POT;
  static constexpr u32 CNT_SIZE = 256;

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    u32 levels;
    u32 dim;
    u32 op;
  };
  static HiZPass *create(RenderingContext *rctx) {
    HiZPass *o = new HiZPass;
    o->init(rctx);
    return o;
  }
  void init(RenderingContext *rctx) {
    this->rctx = rctx;
    auto dev   = rctx->factory;
    timestamps.init(dev);
    hiz = [=] {
      rd::Image_Create_Info info{};
      u32 usage = (u32)rd::Image_Usage_Bits::USAGE_UAV | (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                  (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                  (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC;
      info.format = rd::Format::R16_FLOAT;
      // fixed size depth pyramid
      info.width      = SIZE * 2;
      info.height     = SIZE;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.usage_bits = usage;
      return dev->create_image(info);
    }();
    cnt_image = [&] {
      rd::Image_Create_Info info{};
      u32                   usage = (u32)rd::Image_Usage_Bits::USAGE_UAV;
      info.format                 = rd::Format::R16_UINT;
      // fixed size depth pyramid
      info.width      = CNT_SIZE;
      info.height     = CNT_SIZE;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.usage_bits = usage;
      return dev->create_image(info);
    }();
    signature = [=] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});

      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    reset_cnt_pso = [&] {
      Resource_ID cs{};
      defer(dev->release_resource(cs));
      return dev->create_compute_pso(signature,
                                     cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(1, 0)]] globallycoherent RWTexture2D<uint>   counter_image : register(u1, space0);

[numthreads(32, 32, 1)] void main(uint3 tid : SV_DispatchThreadID) {
  int2 size;
  counter_image.GetDimensions(size.x, size.y);
  if (all(tid.xy < size))
    counter_image[tid.xy] = 0;
}
)"),
                                                             NULL, 0));
    }();

    pso = [&] {
      Resource_ID cs{};
      defer(dev->release_resource(cs));
      Pair<string_ref, string_ref> defines[] = {{stref_s("MAX_OP"), stref_s("")}};
      return dev->create_compute_pso(signature,
                                     cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
struct PushConstants {
  u32 levels;
  u32 dim;
  u32 op;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

#define OP_MAX 1
#define OP_MIN 2

[[vk::binding(0, 0)]] Texture2D<float4>                    src_tex       : register(t0, space0);
[[vk::binding(1, 0)]] globallycoherent RWTexture2D<uint>   counter_image : register(u1, space0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<float4> tex           : register(u2, space0);
groupshared u32                                            group_cnt;

// https://twitter.com/SebAaltonen/status/1328780529790033921
// Single pass mip generator
uint4 calculate_mip_rect(uint dimensions, uint mip)
{
    uint pixels_mip = dimensions >> mip;
    uint4 uv_rect = uint4(0, 0, pixels_mip, pixels_mip);
    if (mip > 0)
    {
        uv_rect.x = dimensions;
        uv_rect.y = dimensions - pixels_mip * 2;
    }
    return uv_rect;
}

[numthreads(8, 8, 1)] void main(uint3 tid
                                  : SV_DispatchThreadID, uint3 xy_local
                                  : SV_GroupThreadID, uint3    xy_group
                                  : SV_GroupID) {
  int2 src_size, dst_size;
  src_tex.GetDimensions(src_size.x, src_size.y);
  tex.GetDimensions(dst_size.x, dst_size.y);
  {
    //if (all(tid.xy < dst_size)) {
      // Copy top level
      float2 uv   = (float2(tid.xy) + float2(0.5f, 0.5f)) / float2(dst_size.y, dst_size.y);
      tex[tid.xy] = src_tex.Load(int3((float2(src_size) * uv), 0));
    //}
    GroupMemoryBarrierWithGroupSync();
  }
  for (uint mip = 0; mip < pc.levels; mip++)
  {
    uint2 xy = xy_group.xy * 8 + xy_local.xy;
 
    uint4 src_rect = calculate_mip_rect(pc.dim, mip);
    uint4 dst_rect = calculate_mip_rect(pc.dim, mip + 1);

    uint2 xy2 = 2 * xy;
    float z00 = tex.Load(int3(src_rect.xy + xy2 + uint2(0, 0), 0)).x;
    float z01 = tex.Load(int3(src_rect.xy + xy2 + uint2(0, 1), 0)).x;
    float z10 = tex.Load(int3(src_rect.xy + xy2 + uint2(1, 0), 0)).x;
    float z11 = tex.Load(int3(src_rect.xy + xy2 + uint2(1, 1), 0)).x;

    #ifdef MAX_OP
      float z_min = max(max(z00, z01), max(z10, z11));
    #else
      float z_min = min(min(z00, z01), min(z10, z11));
    #endif
    tex[int2(dst_rect.xy + xy)] = float4(z_min, 0.0f, 0.0f, 0.0f);

    // Barrier + coherent attribute guarantee that other thread groups see the data
    GroupMemoryBarrierWithGroupSync();
        
    // There are 2x2 groups that are candidates for downsampling the same 2x2 lower resolution region
    // We take the last one to finish, since this guarantees that the next level input data is finished
    xy_group.xy /= 2;
 
    // First lane in group does the atomic increment
    // We bitpack sixteen 2 bit counters to a single 32 bit counter to avoid clearing the counter (double barrier)
    if (all(xy_local.xy == uint2(0,0)))
    {
      u32 cnt;
      InterlockedAdd(counter_image[int2(xy_group.xy)], 1 << (mip*2), cnt);
      group_cnt = cnt;
    }
 
    // For each 2x2 group tile, the last one to finish lives to the next mip level. Other 3 die.
    // This branch is guaranteed to kill the whole group, as it's based on groupshared value
    if (((group_cnt >> (mip*2)) & 3) < 3) return;
  }
}

)"),
                                                             defines, 1));
    }();
  }
  void render() {
    auto dev = rctx->factory;
    timestamps.update(dev);
    u32         width    = rctx->config->get_u32("g_buffer_width");
    u32         height   = rctx->config->get_u32("g_buffer_height");
    Resource_ID depth_rt = rctx->pass_mng->getPass(GBufferPass::NAME)->getBuffer("GBuffer.Depth");

    rd::IBinding_Table *table = dev->create_binding_table(signature);
    defer(table->release());

    rd::ICtx *ctx = dev->start_compute_pass();

    {
      TracyVulkIINamedZone(ctx, "Build HiZ");
      timestamps.begin_range(ctx);
      auto desc = dev->get_image_info(depth_rt);

      // timestamp.begin_range(ctx);
      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      ctx->image_barrier(hiz, rd::Image_Access::UAV);
      ctx->image_barrier(cnt_image, rd::Image_Access::UAV);
      ctx->image_barrier(depth_rt, rd::Image_Access::SAMPLED);

      table->bind_UAV_texture(0, 2, 0, hiz, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
      table->bind_texture(0, 0, 0, depth_rt, rd::Image_Subresource::top_level(0),
                          rd::Format::R32_FLOAT);
      table->bind_UAV_texture(0, 1, 0, cnt_image, rd::Image_Subresource::top_level(),
                              rd::Format::NATIVE);

      ctx->bind_table(table);
      ctx->bind_compute(reset_cnt_pso);
      ctx->dispatch(CNT_SIZE / 32, CNT_SIZE / 32, 1);
      ctx->image_barrier(cnt_image, rd::Image_Access::UAV);
      ctx->bind_compute(pso);
      PushConstants pc{};
      pc.levels = SIZE_POT;
      pc.op     = 0;
      pc.dim    = SIZE;
      table->push_constants(&pc, 0, sizeof(pc));
      ctx->dispatch(SIZE / 8, SIZE / 8, 1);

      timestamps.end_range(ctx);
    }
    Resource_ID e = dev->end_compute_pass(ctx);
    timestamps.commit(e);
  }

  char const *getName() override { return NAME; }
  u32         getNumBuffers() override { return 1; }
  char const *getBufferName(u32 i) override { return "HiZ"; }
  Resource_ID getBuffer(u32 i) override { return hiz; }
  double      getLastDurationInMs() override { return timestamps.duration; }
};

class ComposePass : public IPass {
  ~ComposePass() = default;

  public:
  static constexpr char const *NAME = "Compose Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(sampler_state);                                                                         \
  RESOURCE(rt);                                                                                    \
  RESOURCE(pso);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE
  void release() override {
    timestamps.release(rctx->factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) rctx->factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
    delete this;
  }
#undef RESOURCE_LIST
  RenderingContext *rctx   = NULL;
  u32               width  = 0;
  u32               height = 0;

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {};
  static ComposePass *create(RenderingContext *rctx) {
    ComposePass *o = new ComposePass;
    o->init(rctx);
    return o;
  }
  void init(RenderingContext *rctx) {
    this->rctx = rctx;
    auto dev   = rctx->factory;
    timestamps.init(dev);

    signature = [=] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      set_info.bindings.push({rd::Binding_t::TEXTURE, 16});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = 0; // sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    sampler_state = [&] {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::LINEAR;
      info.max_lod        = 1000.0f;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      return dev->create_sampler(info);
    }();
    pso = [&] {
      Resource_ID cs{};
      defer(dev->release_resource(cs));
      return dev->create_compute_pso(signature,
                                     cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWTexture2D<float4> compose        : register(u0, space0);
[[vk::binding(1, 0)]] SamplerState        ss             : register(s1, space0);

[[vk::binding(2, 0)]] Texture2D<float4>   inputs[16] : register(t2, space0);

#define GBUFFER_NORMAL 0
#define GBUFFER_DEPTH 1
#define GIZMO_LAYER 2

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  uint width, height;
  compose.GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height)
    return;
  float2 uv = (float2(tid.xy) + float2(0.5f, 0.5f)) / float2(width, height);
  float3 normal = inputs[GBUFFER_NORMAL].Load(int3(tid.xy, 0)).xyz;
  float4 gizmo  = inputs[GIZMO_LAYER].Load(int3(tid.xy, 0)).xyzw;
  float3 color = float3_splat(max(0.0f, dot(normal, normalize(float3(1.0, 1.0, 1.0)))));
  compose[tid.xy] = float4(lerp(pow(color, 1.0f), gizmo.xyz, gizmo.w), 1.0f);
}
)"),
                                                             NULL, 0));
    }();
  }
  void update_frame_buffer(RenderingContext *rctx) {
    auto dev = rctx->factory;
    if (rt.is_valid()) dev->release_resource(rt);
    rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits =                          //
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
          (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
  }
  void render() {
    auto dev = rctx->factory;
    timestamps.update(dev);
    u32         width     = rctx->config->get_u32("g_buffer_width");
    u32         height    = rctx->config->get_u32("g_buffer_height");
    Resource_ID depth_rt  = rctx->pass_mng->getPass(GBufferPass::NAME)->getBuffer("GBuffer.Depth");
    Resource_ID normal_rt = rctx->pass_mng->getPass(GBufferPass::NAME)->getBuffer("GBuffer.Normal");
    Resource_ID gizmo_layer = rctx->pass_mng->getPass(GizmoPass::NAME)->getBuffer((u32)0);
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    rd::IBinding_Table *table = dev->create_binding_table(signature);
    defer(table->release());
    table->bind_texture(0, 2, 0, normal_rt, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    table->bind_texture(0, 2, 1, depth_rt, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    table->bind_texture(0, 2, 2, gizmo_layer, rd::Image_Subresource::top_level(),
                        rd::Format::NATIVE);
    table->bind_UAV_texture(0, 0, 0, rt, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    table->bind_sampler(0, 1, sampler_state);
    rd::ICtx *ctx = dev->start_compute_pass();
    ctx->image_barrier(normal_rt, rd::Image_Access::SAMPLED);
    ctx->image_barrier(depth_rt, rd::Image_Access::SAMPLED);
    ctx->image_barrier(gizmo_layer, rd::Image_Access::SAMPLED);
    {
      TracyVulkIINamedZone(ctx, NAME);
      timestamps.begin_range(ctx);
      ctx->bind_compute(pso);
      ctx->bind_table(table);
      ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
      timestamps.end_range(ctx);
    }
    Resource_ID e = dev->end_compute_pass(ctx);
    timestamps.commit(e);
  }

  char const *getName() override { return NAME; }
  u32         getNumBuffers() override { return 1; }
  char const *getBufferName(u32 i) override { return "Compose.RT"; }
  Resource_ID getBuffer(u32 i) override { return rt; }
  double      getLastDurationInMs() override { return timestamps.duration; }
};

#if 1

class Event_Consumer : public IGUIApp, public IPassMng {
  public:
  Hash_Table<u64, float4>     minmax{};
  InlineArray<IPass *, 0x100> passes{};
  // GBufferPass gbuffer_pass;
  // GizmoPass   gizmo_pass;
  // ComposePass compose_pass;
  IPass *getPass(char const *name) override {
    ito(passes.size) if (strcmp(passes[i]->getName(), name) == 0) return passes[i];
    return NULL;
  }
  RenderingContext *rctx = NULL;
  void              init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("camera")) {
        rctx->gizmo_layer->get_camera().traverse(l->next);
      } else if (l->cmp_symbol("config")) {
        rctx->config->traverse(l->next);
      } else if (l->cmp_symbol("scene")) {
        rctx->scene->restore(l);
      }
    }
  }
  void on_gui() override { //
    timer.update();
    ImGui::Begin("Scene");
    {
      String_Builder sb;
      sb.init();
      defer(sb.release());
      rctx->scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        rctx->scene->restore(cur);
      }
    }
    ImGui::End();

    ImGui::Begin("Config");
    if (rctx->config->on_imgui()) rctx->dump();
    ito(passes.size) {
      ImGui::Text("%s %fms", passes[i]->getName(), passes[i]->getLastDurationInMs());
    }

    if (ImGui::Button("Rebuild BVH")) {
      rctx->scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (mn->getComponent<BVHSufraceComponent>()) {
            mn->getComponent<BVHSufraceComponent>()->updateBVH();
          }
        }
      });
    }
    ImGui::End();

    ito(passes.size) {
      jto(passes[i]->getNumBuffers()) {
        ImGui::Begin(passes[i]->getBufferName(j));
        {
          rctx->gizmo_layer->per_imgui_window();
          auto    wsize = get_window_size();
          float4 *m     = minmax.get_or_null(passes[i]->getBuffer(i).data);
          if (m == NULL)
            minmax.insert(passes[i]->getBuffer(i).data, float4{0.0f, 1.0f, 0.0f, 0.0f});
          m = minmax.get_or_null(passes[i]->getBuffer(i).data);
          ImGui::SetNextItemWidth(100);
          ImGui::DragFloat("min", &m->x, 1.0e-3f);
          ImGui::SameLine();
          ImGui::SetNextItemWidth(100);
          ImGui::DragFloat("max", &m->y, 1.0e-3f);
          ImGui::Image(bind_texture(passes[i]->getBuffer(i), 0, 0, rd::Format::NATIVE, m->x, m->y),
                       ImVec2(wsize.x, wsize.y - 20.0f));
          { Ray ray = rctx->gizmo_layer->getMouseRay(); }
        }
        ImGui::End();
      }
      ImGui::Text("%s %fms", passes[i]->getName(), passes[i]->getLastDurationInMs());
    }
  }
  void on_init() override { //
    minmax.init();
    rctx          = new RenderingContext;
    rctx->factory = this->factory;
    TMP_STORAGE_SCOPE;

    // new XYZDragGizmo(gizmo_layer, &pos);
    rctx->scene  = Scene::create();
    rctx->config = new Config;
    rctx->config->init(stref_s(R"(
 (
  (add u32  g_buffer_width 512 (min 4) (max 2048))
  (add u32  g_buffer_height 512 (min 4) (max 2048))
  (add u32  baking.size 512 (min 4) (max 4096))
  (add bool G.I.color_triangles 0)
 )
 )"));
    rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/ssr_test.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/norradalur-froyar/scene.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/cut.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/untitled.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
    rctx->scene->update();
    rctx->scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        GfxSufraceComponent::create(rctx->factory, mn);
        // MeshletSufraceComponent::create(mn, 255, 256);
        // GfxMeshletSufraceComponent::create(factory, mn);
      }
    });
    rctx->pass_mng = this;
    passes.push(GBufferPass::create(rctx));
    passes.push(GizmoPass::create(rctx));
    passes.push(HiZPass::create(rctx));
    passes.push(ComposePass::create(rctx));

    rctx->gizmo_layer = Gizmo_Layer::create(factory, ((GizmoPass *)getPass(GizmoPass::NAME))->pass);
    char *state       = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
  }
  void on_release() override { //
    minmax.release();
    rctx->dump();
    rctx->gizmo_layer->release();
    rctx->scene->release();
    rctx->config->release();
    ito(passes.size) passes[i]->release();
    passes.release();
    delete rctx->config;
    delete rctx;
  }
  void on_frame() override { //
    rctx->scene->get_root()->update();
    ito(passes.size) passes[i]->render();
  }
};
#endif
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto window_loop = [](rd::Impl_t impl) { IGUIApp::start<Event_Consumer>(impl); };
  // std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  // std::thread dx12_thread = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  // vulkan_thread.join();
  // dx12_thread.join();

  // window_loop(rd::Impl_t::VULKAN);
  window_loop(rd::Impl_t::DX12);
  return 0;
}