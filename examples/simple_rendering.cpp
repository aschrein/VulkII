#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC
//#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <atomic>
//#include <functional>
#include <3rdparty/half.hpp>
#include <condition_variable>
#include <imgui.h>
#include <mutex>
#include <thread>

#include <embree3/rtcore_builder.h>

class BufferThing {
#define RESOURCE_LIST                                                                              \
  RESOURCE(cs0);                                                                                   \
  RESOURCE(cs1);                                                                                   \
  RESOURCE(cs2);                                                                                   \
  RESOURCE(buffer);                                                                                \
  RESOURCE(buffer1);                                                                               \
  RESOURCE(readback);                                                                              \
  RESOURCE(signature);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  public:
  void init(rd::IDevice *dev) {
    buffer = [dev] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = sizeof(u32) * 16 * 1024 * 1024;
      return dev->create_buffer(buf_info);
    }();
    // Allocate a buffer.
    buffer1 = [dev] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = sizeof(u32) * 16 * 1024 * 1024;
      return dev->create_buffer(buf_info);
    }();

    signature = [dev] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
      set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = 4;
      return dev->create_signature(table_info);
    }();

    cs0 = dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x % 1024;
    for (uint i = 0; i < 100; i++)
      BufferOut.Store<uint>(id * 4, BufferOut.Load<uint>(id * 4) + 1);
}
)"),
                                                                NULL, 0));
    cs1 = dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x % 1024;
    for (uint i = 0; i < 100; i++)
      BufferOut.Store<uint>(id * 4, BufferOut.Load<uint>(id * 4) + 1);
}
)"),
                                                                NULL, 0));
    cs2 = dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut: register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);
  
struct CullPushConstants
{
  uint val;
};
[[vk::push_constant]] ConstantBuffer<CullPushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;
  
[numthreads(64, 1, 1)]
  void main(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x % 1024;
    for (uint i = 0; i < 100; i++)
      BufferOut.Store<uint>(id * 4, BufferOut.Load<uint>(id * 4) * pc.val);
}
)"),
                                                                NULL, 0));
  }
  void test_buffers(rd::IDevice *dev) {
    Resource_ID wevent_0{};
    Resource_ID wevent_1{};
    {
      rd::ICtx *ctx = dev->start_async_compute_pass();
      {
        TracyVulkIINamedZone(ctx, "Async Compute Example 1");
        ctx->bind_compute(cs0);
        rd::IBinding_Table *table = dev->create_binding_table(signature);
        defer(table->release());
        table->bind_UAV_buffer(0, 0, buffer, 0, sizeof(u32) * 1024);
        table->bind_UAV_buffer(0, 1, buffer, 0, sizeof(u32) * 1024);
        ctx->bind_table(table);
        ctx->dispatch(1024 * 32, 1, 1);
      }
      wevent_0 = dev->end_async_compute_pass(ctx);
    }
    // dev->wait_idle();
    u32 val = 2;
    {
      rd::ICtx *ctx = dev->start_async_compute_pass();
      {
        TracyVulkIINamedZone(ctx, "Async Compute Example 2");
        // ctx->wait_for_event(wevent_0);
        ctx->bind_compute(cs1);
        rd::IBinding_Table *table = dev->create_binding_table(signature);
        defer(table->release());
        table->bind_UAV_buffer(0, 0, buffer1, 0, sizeof(u32) * 1024);
        table->bind_UAV_buffer(0, 1, buffer1, 0, sizeof(u32) * 1024);
        ctx->bind_table(table);
        // ctx->buffer_barrier(buffer, rd::Buffer_Access::UAV);
        ctx->dispatch(1024 * 32, 1, 1);
      }
      wevent_1 = dev->end_async_compute_pass(ctx);
    }
    {
      rd::ICtx *ctx = dev->start_async_copy_pass();
      {
        // ctx->wait_for_event(wevent_0);
        // ctx->wait_for_event(wevent_1);
        TracyVulkIINamedZone(ctx, "Async Copy Example");
        ctx->copy_buffer(buffer, 0, buffer1, 0, sizeof(u32) * 16 * 1024 * 1024);
      }
      wevent_1 = dev->end_async_copy_pass(ctx);
    }
  }
  void release(rd::IDevice *factory) {
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};

#if 1
struct RenderingContext {
  rd::IDevice *factory     = NULL;
  Config *     config      = NULL;
  Scene *      scene       = NULL;
  Gizmo_Layer *gizmo_layer = NULL;
};
class GBufferPass {
  public:
#  define RESOURCE_LIST                                                                            \
    RESOURCE(signature);                                                                           \
    RESOURCE(pso);                                                                                 \
    RESOURCE(pass);                                                                                \
    RESOURCE(frame_buffer);                                                                        \
    RESOURCE(normal_rt);                                                                           \
    RESOURCE(depth_rt);                                                                            \
    RESOURCE(gbuffer_vs);                                                                          \
    RESOURCE(gbuffer_ps);

#  define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#  undef RESOURCE

  u32         width  = 0;
  u32         height = 0;
  //BufferThing bthing{};

  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};

  public:
  // TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
    //bthing.init(rctx.factory);
    gbuffer_vs = rctx.factory->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  //[[vk::location(2)]] float2 uv      : TEXCOORD1;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] float3 normal  : NORMAL;
  //[[vk::location(4)]] float2 uv      : TEXCOORD0;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.normal = mul(pc.world_transform, float4(input.normal.xyz, 0.0f)).xyz;
  //output.uv     = input.uv;
  output.pos    = mul(pc.viewproj, mul(pc.world_transform, float4(input.pos, 1.0f)));
  return output;
}
)"),
                                             NULL, 0);
    gbuffer_ps = rctx.factory->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  //[[vk::location(2)]] float2 uv      : TEXCOORD1;
};

float4 main(in PSInput input) : SV_TARGET0 {
  return float4_splat(
          abs(
              dot(
                input.normal,
                normalize(float3(1.0, 1.0, 1.0))
              )
            )
         );
}
)"),
                                             NULL, 0);
    signature  = [=] {
      rd::Binding_Space_Create_Info set_info{};
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return rctx.factory->create_signature(table_info);
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
      info.depth_target.format            = rd::Format::D32_FLOAT;
      return rctx.factory->create_render_pass(info);
    }();

    pso = [=] {
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state{};
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
      /*{
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 2;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 2;
        info.offset   = 0;
        info.type     = rd::Attriute_t::TEXCOORD0;
        gfx_state.IA_set_attribute(info);
      }*/
      gfx_state.IA_set_vertex_binding(0, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, 12, rd::Input_Rate::VERTEX);
      // gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return rctx.factory->create_graphics_pso(signature, pass, gfx_state);
    }();
    rctx.scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GfxSufraceComponent>() == NULL) {
          GfxSufraceComponent::create(rctx.factory, mn);
          // mn->getComponent<GfxSufraceComponent>()->buildBVH();
        }

        // render_bvh(bvh_visualizer_offset, mn->getComponent<GfxSufraceComponent>()->getBVH(),
        // rctx.gizmo_layer);
      }
    });
  }
  void update_frame_buffer(RenderingContext rctx) {
    if (frame_buffer.is_valid()) rctx.factory->release_resource(frame_buffer);
    if (normal_rt.is_valid()) rctx.factory->release_resource(normal_rt);
    if (depth_rt.is_valid()) rctx.factory->release_resource(depth_rt);

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
      return rctx.factory->create_image(rt0_info);
    }();
    depth_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::D32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT;
      return rctx.factory->create_image(rt0_info);
    }();
    frame_buffer = [=] {
      rd::Frame_Buffer_Create_Info info{};
      rd::RT_View                  rt0{};
      rt0.image  = normal_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);

      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_FLOAT;
      return rctx.factory->create_frame_buffer(pass, info);
    }();
  }
  void render(RenderingContext rctx) {
    // timestamps.update(rctx.factory);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f,
    // 0.0f));
    //bthing.test_buffers(rctx.factory);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }
    if (rctx.config->get_bool("render_gizmo")) {
      auto g_camera = rctx.gizmo_layer->get_camera();
      {
        float dx = 1.0e-1f * g_camera.distance;
        rctx.gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
        rctx.gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{dx, 0.0f, 0.0f},
                                        dx * 0.04f, float3{1.0f, 0.0f, 0.0f});
        rctx.gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, dx, 0.0f},
                                        dx * 0.04f, float3{0.0f, 1.0f, 0.0f});
        rctx.gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, 0.0f, dx},
                                        dx * 0.04f, float3{0.0f, 0.0f, 1.0f});
      }
    }

    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();

    rd::ICtx *ctx = rctx.factory->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "GBuffer Pass");
      ctx->start_render_pass();

      // timestamps.insert(rctx.factory, ctx);
      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      float dx = 0.0f;
      // float4x4 world = float4x4(  //
      //    1.0f, 0.0f, 0.0f, dx,   //
      //    0.0f, 1.0f, 0.0f, 0.0f, //
      //    0.0f, 0.0f, 1.0f, 0.0f, //
      //    0.0f, 0.0f, 0.0f, 1.0f  //
      //);
      pc.viewproj = viewproj;

      rd::IBinding_Table *table = rctx.factory->create_binding_table(signature);
      defer(table->release());
      table->push_constants(&viewproj, 0, sizeof(float4x4));
      ctx->bind_table(table);
      ctx->bind_graphics_pso(pso);
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
          float4x4             world = mn->get_transform();
          pc.world_transform         = world;
          table->push_constants(&pc, 0, sizeof(pc));
          ito(gs->getNumSurfaces()) {
            GfxSurface *s = gs->getSurface(i);
            s->draw(ctx, gfx_state);
          }
        }
      });
      rctx.gizmo_layer->render(ctx, width, height);
      ctx->end_render_pass();
    }
    rctx.factory->end_render_pass(ctx);

    // timestamps.insert(rctx.factory, ctx);

    // rctx.gizmo_layer->reset();
    // fprintf(stdout, "[END FRAME]\n");
    // fflush(stdout);
    // for (auto &th : threads) th.join();
    // threads.clear();
  }
  void release(rd::IDevice *factory) {
    //bthing.release(factory);
#  define RESOURCE(name)                                                                           \
    if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#  undef RESOURCE
  }
#  undef RESOURCE_LIST
};
#endif
#if 1
class Event_Consumer : public IGUIApp {
  public:
  GBufferPass      gbuffer_pass;
  RenderingContext rctx{};
  void             init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("camera")) {
        rctx.gizmo_layer->get_camera().traverse(l->next);
      } else if (l->cmp_symbol("config")) {
        rctx.config->traverse(l->next);
      } else if (l->cmp_symbol("scene")) {
        rctx.scene->restore(l);
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
      rctx.scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        rctx.scene->restore(cur);
      }
    }
    ImGui::End();

    ImGui::Begin("Config");
    rctx.config->on_imgui();
    // ImGui::Text("%fms", gbuffer_pass.timestamps.duration);
    ImGui::End();

    ImGui::Begin("main viewport");
    rctx.gizmo_layer->per_imgui_window();
    auto wsize = get_window_size();
    ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                 ImVec2(wsize.x, wsize.y));
    { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    ImGui::End();
  }
  void on_init() override { //

    rctx.factory = this->factory;
    TMP_STORAGE_SCOPE;

    // new XYZDragGizmo(gizmo_layer, &pos);
    rctx.scene  = Scene::create();
    rctx.config = new Config;
    rctx.config->init(stref_s(R"(
 (
  (add u32  g_buffer_width 512 (min 4) (max 1024))
  (add u32  g_buffer_height 512 (min 4) (max 1024))
 )
 )"));
    rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
    rctx.scene->update();
    gbuffer_pass.init(rctx);
    rctx.gizmo_layer = Gizmo_Layer::create(factory, gbuffer_pass.pass);
    char *state      = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
  }
  void on_release() override { //
    FILE *scene_dump = fopen("scene_state", "wb");
    fprintf(scene_dump, "(\n");
    defer(fclose(scene_dump));
    rctx.gizmo_layer->get_camera().dump(scene_dump);
    rctx.config->dump(scene_dump);
    {
      String_Builder sb;
      sb.init();
      rctx.scene->save(sb);
      fwrite(sb.get_str().ptr, 1, sb.get_str().len, scene_dump);
      sb.release();
    }
    fprintf(scene_dump, ")\n");
    rctx.gizmo_layer->release();
    rctx.scene->release();
    delete rctx.config;
  }
  void on_frame() override { //
    /*rctx.scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GizmoComponent>() == NULL) {
          GizmoComponent::create(rctx.gizmo_layer, mn);
        }
      }
    });*/
    rctx.scene->get_root()->update();
    gbuffer_pass.render(rctx);
  }
};
#endif
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto window_loop = [](rd::Impl_t impl) { IGUIApp::start<Event_Consumer>(impl); };
  // std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  std::thread dx12_thread = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  // vulkan_thread.join();
  dx12_thread.join();

  return 0;
}
