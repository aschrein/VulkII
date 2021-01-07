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

#if 1
struct RenderingContext {
  rd::IDevice *factory = NULL;
  Config *     config  = NULL;
  Scene *      scene   = NULL;
  // Gizmo_Layer * gizmo_layer = NULL;
};
Camera g_camera;
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

  u32 width  = 0;
  u32 height = 0;

  void render_once(rd::ICtx *ctx, Scene *scene, float4x4 viewproj, float4x4 world) {

    /* if (rctx.config->get_bool("render_scene_wireframe")) {
       ctx->PS_set_shader(rctx.factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
 struct PSInput {
   [[vk::location(0)]] float4 pos     : SV_POSITION;
   [[vk::location(1)]] float3 normal  : TEXCOORD0;
   [[vk::location(2)]] float2 uv      : TEXCOORD1;
 };

 float4 main(in PSInput input) : SV_TARGET0 {
   return float4_splat(0.0f);
 }
 )"),
                                                          NULL, 0));
       rctx.scene->traverse([&](Node *node) {
         if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
           GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
           float4x4             model = mn->get_transform();
           ctx->push_constants(&model, 64, sizeof(model));
           ito(gs->getNumSurfaces()) {
             GfxSurface *s = gs->getSurface(i);
             s->draw(ctx, attribute_to_location);
           }
         }
       });
     }*/
  }

  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};

  public:
  // TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
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
      //gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
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

    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }
    // if (rctx.config->get_bool("render_gizmo")) {
    //  auto g_camera = rctx.gizmo_layer->get_camera();
    //  {
    //    float dx = 1.0e-1f * g_camera.distance;
    //    rctx.gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
    //    rctx.gizmo_layer->draw_cylinder(g_camera.look_at,
    //                                    g_camera.look_at + float3{dx, 0.0f, 0.0f}, dx * 0.04f,
    //                                    float3{1.0f, 0.0f, 0.0f});
    //    rctx.gizmo_layer->draw_cylinder(g_camera.look_at,
    //                                    g_camera.look_at + float3{0.0f, dx, 0.0f}, dx * 0.04f,
    //                                    float3{0.0f, 1.0f, 0.0f});
    //    rctx.gizmo_layer->draw_cylinder(g_camera.look_at,
    //                                    g_camera.look_at + float3{0.0f, 0.0f, dx}, dx * 0.04f,
    //                                    float3{0.0f, 0.0f, 1.0f});
    //  }
    //}

    //{
    //  int          i   = 0;
    //  rd::ICtx *ctx = rctx.factory->start_render_pass(info);

    //  float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();
    //  // timestamps.insert(rctx.factory, ctx);
    //  ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    //  ctx->set_scissor(0, 0, width, height);
    //  float    dx    = 1.0f;
    //  float4x4 model = float4x4(    //
    //      1.0f, 0.0f, 0.0f, dx * i, //
    //      0.0f, 1.0f, 0.0f, 0.0f,   //
    //      0.0f, 0.0f, 1.0f, 0.0f,   //
    //      0.0f, 0.0f, 0.0f, 1.0f    //
    //  );
    //  render_once(ctx, rctx.scene, viewproj, model);
    //  if (i == 0) {
    //    rctx.gizmo_layer->render(rctx.factory, ctx, width, height);
    //  }
    //  rctx.factory->end_render_pass(ctx);
    //}
    // fprintf(stdout, "[START FRAME]\n");
    // fflush(stdout);

    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = g_camera.viewproj();

    rd::ICtx *ctx = rctx.factory->start_render_pass(pass, frame_buffer);
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
        pc.world_transform         = transpose(world);
        table->push_constants(&pc, 0, sizeof(pc));
        ito(gs->getNumSurfaces()) {
          GfxSurface *s = gs->getSurface(i);
          s->draw(ctx, gfx_state);
        }
      }
    });

    // rctx.gizmo_layer->render(rctx.factory, ctx, width, height);
    ctx->end_render_pass();
    rctx.factory->end_render_pass(ctx);

    // timestamps.insert(rctx.factory, ctx);

    // rctx.gizmo_layer->reset();
    // fprintf(stdout, "[END FRAME]\n");
    // fflush(stdout);
    // for (auto &th : threads) th.join();
    // threads.clear();
  }
  void release(rd::IDevice *factory) {
#  define RESOURCE(name)                                                                           \
    if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#  undef RESOURCE
  }
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
      /* if (l->cmp_symbol("camera")) {
         rctx.gizmo_layer->get_camera().traverse(l->next);
       } else if (l->cmp_symbol("config")) {
         rctx.config->traverse(l->next);
       } else if (l->cmp_symbol("scene")) {
         rctx.scene->restore(l);
       }*/
    }
  }
  void on_gui() override { //

    // bool show = true;
    // ShowExampleAppCustomNodeGraph(&show);
    // ImGui::TestNodeGraphEditor();
    // ImGui::Begin("Text");
    // te.Render("Editor");
    // ImGui::End();
    {
      static int2 mpos      = {};
      static int2 last_mpos = {};
      ImVec2      imguimpos = ImGui::GetMousePos();
      auto        wpos      = ImGui::GetCursorScreenPos();
      auto        wsize     = ImGui::GetWindowSize();
      g_camera.aspect       = float(wsize.x) / wsize.y;
      imguimpos.x -= wpos.x;
      imguimpos.y -= wpos.y;
      last_mpos     = mpos;
      mpos          = int2(imguimpos.x, imguimpos.y);
      i32  dx       = mpos.x - last_mpos.x;
      i32  dy       = mpos.y - last_mpos.y;
      auto scroll_y = ImGui::GetIO().MouseWheel;
      if (scroll_y) {
        g_camera.distance += g_camera.distance * 2.e-1 * scroll_y;
        g_camera.distance = clamp(g_camera.distance, 1.0e-3f, 1000.0f);
      }
      f32 camera_speed = 2.0f * g_camera.distance;
      if (ImGui::GetIO().KeysDown[SDL_SCANCODE_LSHIFT]) {
        camera_speed = 10.0f * g_camera.distance;
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
        g_camera.look_at += glm::normalize(camera_diff) * camera_speed * (float)timer.dt;
      }
      if (ImGui::IsMouseDown(0)) {
        g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
        g_camera.theta -= (float)(dy)*5.0e-3f;
      }
    }
    timer.update();
    g_camera.update();
    ImGui::Begin("Scene");
    {
      String_Builder sb;
      sb.init();
      defer(sb.release());
      // rctx.scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        // rctx.scene->restore(cur);
      }
    }
    ImGui::End();

    ImGui::Begin("Config");
    // rctx.config->on_imgui();
    // ImGui::Text("%fms", gbuffer_pass.timestamps.duration);
    ImGui::End();

    ImGui::Begin("main viewport");
    // rctx.gizmo_layer->per_imgui_window();
    auto wsize = get_window_size();
    ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                 ImVec2(wsize.x, wsize.y));
    //{ Ray ray = rctx.gizmo_layer->getMouseRay(); }
    ImGui::End();
  }
  void on_init() override { //
    g_camera.init();
    rctx.factory = this->factory;
    TMP_STORAGE_SCOPE;
    // rctx.gizmo_layer = Gizmo_Layer::create(factory);
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
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }

    gbuffer_pass.init(rctx);
  }
  void on_release() override { //
    g_camera.release();
    // thread_pool.release();
    FILE *scene_dump = fopen("scene_state", "wb");
    fprintf(scene_dump, "(\n");
    defer(fclose(scene_dump));
    // rctx.gizmo_layer->get_camera().dump(scene_dump);
    // rctx.config->dump(scene_dump);
    {
      String_Builder sb;
      sb.init();
      // rctx.scene->save(sb);
      fwrite(sb.get_str().ptr, 1, sb.get_str().len, scene_dump);
      sb.release();
    }
    fprintf(scene_dump, ")\n");
    // rctx.gizmo_layer->release();
    // rctx.scene->release();
    // delete rctx.config;
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

  auto        window_loop   = [](rd::Impl_t impl) { IGUIApp::start<Event_Consumer>(impl); };
  std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  std::thread dx12_thread   = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  vulkan_thread.join();
  dx12_thread.join();

  return 0;
}
