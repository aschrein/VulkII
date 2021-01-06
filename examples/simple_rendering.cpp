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

#if 0
struct RenderingContext {
  rd::IDevice *factory     = NULL;
  Config *      config      = NULL;
  Scene *       scene       = NULL;
  Gizmo_Layer * gizmo_layer = NULL;
};
class GBufferPass {
  public:
  Resource_ID normal_rt{};
  Resource_ID depth_rt{};
  Resource_ID gbuffer_vs{};
  Resource_ID gbuffer_ps{};

  void render_once(rd::ICtx *ctx, Scene *scene, float4x4 viewproj, float4x4 world) {
    ctx->push_constants(&viewproj, 0, sizeof(float4x4));
    setup_default_state(ctx, 1);
    rd::DS_State ds_state;
    rd::RS_State rs_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);

    ctx->VS_set_shader(gbuffer_vs);
    ctx->PS_set_shader(gbuffer_ps);
    static u32 attribute_to_location[] = {
        0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    // if (rctx.config->get_bool("render_scene")) {
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);

    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CCW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    ctx->RS_set_state(rs_state);

    scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
        float4x4             model = world * mn->get_transform();
        ctx->push_constants(&model, 64, sizeof(model));
        ito(gs->getNumSurfaces()) {
          GfxSurface *s = gs->getSurface(i);
          s->draw(ctx, attribute_to_location);
        }
      }
    });
    //}
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::LINE;
    rs_state.front_face   = rd::Front_Face::CCW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    ctx->RS_set_state(rs_state);
    ctx->RS_set_depth_bias(0.1f);
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
  std::vector<std::thread>    threads;
  std::condition_variable     work_finish_cv;
  std::mutex                  work_finish_mutex;
  std::condition_variable     work_start_cv;
  std::mutex                  work_start_mutex;
  std::atomic<bool>           working;
  std::atomic<int>            new_work;
  std::atomic<bool>           one_time_wake[0x100];
  rd::Render_Pass_Create_Info info{};

  public:
  TimeStamp_Pool timestamps = {};

  void init() {}
  void render(RenderingContext rctx) {
    timestamps.update(rctx.factory);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f, 0.0f));
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

    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    {
      rd::Image_Create_Info rt0_info;

      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      normal_rt = get_or_create_image(rctx.factory, rt0_info, normal_rt);
    }
    {
      rd::Image_Create_Info rt0_info;

      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::D32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT;
      depth_rt            = get_or_create_image(rctx.factory, rt0_info, depth_rt);
    }
    // Clear Images
    {
      rd::Render_Pass_Create_Info info{};
      info.width  = width;
      info.height = height;
      rd::RT_View rt0{};
      rt0.image             = normal_rt;
      rt0.format            = rd::Format::NATIVE;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 1.0f;
      info.rts.push(rt0);

      info.depth_target.image             = depth_rt;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::NATIVE;
      auto ctx                            = rctx.factory->start_render_pass(info);
      rctx.factory->end_render_pass(ctx);
    }
    {
      MEMZERO(info);
      info.width  = width;
      info.height = height;
      rd::RT_View rt0;
      MEMZERO(rt0);
      rt0.image             = normal_rt;
      rt0.format            = rd::Format::NATIVE;
      rt0.clear_color.clear = false;
      info.rts.push(rt0);
      info.depth_target.image             = depth_rt;
      info.depth_target.clear_depth.clear = false;
      info.depth_target.format            = rd::Format::NATIVE;

      if (rctx.config->get_bool("render_gizmo")) {
        auto g_camera = rctx.gizmo_layer->get_camera();
        {
          float dx = 1.0e-1f * g_camera.distance;
          rctx.gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
          rctx.gizmo_layer->draw_cylinder(g_camera.look_at,
                                          g_camera.look_at + float3{dx, 0.0f, 0.0f}, dx * 0.04f,
                                          float3{1.0f, 0.0f, 0.0f});
          rctx.gizmo_layer->draw_cylinder(g_camera.look_at,
                                          g_camera.look_at + float3{0.0f, dx, 0.0f}, dx * 0.04f,
                                          float3{0.0f, 1.0f, 0.0f});
          rctx.gizmo_layer->draw_cylinder(g_camera.look_at,
                                          g_camera.look_at + float3{0.0f, 0.0f, dx}, dx * 0.04f,
                                          float3{0.0f, 0.0f, 1.0f});
        }
      }

      if (gbuffer_vs.is_null()) {
        gbuffer_vs = rctx.factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::binding(0, 0)]] ConstantBuffer<PushConstants> pc : register(b0, space0);
//[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(4)]] float2 uv      : TEXCOORD1;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.normal = mul(pc.world_transform, float4(input.normal.xyz, 0.0f)).xyz;
  output.uv     = input.uv;
  output.pos    = mul(pc.viewproj, mul(pc.world_transform, float4(input.pos, 1.0f)));
  return output;
}
)"),
                                                     NULL, 0);
      }
      if (gbuffer_ps.is_null()) {
        gbuffer_ps = rctx.factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
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
      }
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
      //fprintf(stdout, "[START FRAME]\n");
      //fflush(stdout);
      if (threads.size() == 0) {
        working = true;
        ito(2) {
          int _thread_id = i;
          threads.emplace_back(std::thread([this, rctx, _thread_id] {
            int thread_id = _thread_id;
            while (working) {
              u32 width  = rctx.config->get_u32("g_buffer_width");
              u32 height = rctx.config->get_u32("g_buffer_height");

              std::unique_lock<std::mutex> lk(work_start_mutex);
              work_start_cv.wait(lk, [&] { return one_time_wake[thread_id] && new_work != 0; });
              one_time_wake[thread_id] = false;
              //fprintf(stdout, "[START RECORDING] thread %i\n", thread_id);
              //fflush(stdout);
              // if (thread_id == 0) {
              struct PushConstants {
                float4x4 viewproj;
                float4x4 world_transform;
              } pc;

              rd::Buffer_Create_Info buf_info;
              MEMZERO(buf_info);
              buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
              buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER |
                                    (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
              buf_info.size        = sizeof(PushConstants);
              Resource_ID cbuffer  = rctx.factory->create_buffer(buf_info);
              float4x4    viewproj = rctx.gizmo_layer->get_camera().viewproj();
              rctx.factory->release_resource(cbuffer);

              rd::ICtx *ctx = rctx.factory->start_render_pass(info);

              // timestamps.insert(rctx.factory, ctx);
              ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
              ctx->set_scissor(0, 0, width, height);
              float    dx        = 1.0f;
              float4x4 model     = float4x4(            //
                  1.0f, 0.0f, 0.0f, dx * thread_id, //
                  0.0f, 1.0f, 0.0f, 0.0f,           //
                  0.0f, 0.0f, 1.0f, 0.0f,           //
                  0.0f, 0.0f, 0.0f, 1.0f            //
              );
              pc.viewproj        = viewproj;
              pc.world_transform = transpose(model);
              ctx->update_buffer(cbuffer, 0, &pc, sizeof(PushConstants));
              ctx->bind_uniform_buffer(0, 0, cbuffer, 0, sizeof(pc));
              render_once(ctx, rctx.scene, viewproj, transpose(model));
              if (thread_id == 0) {
                rctx.gizmo_layer->render(rctx.factory, ctx, width, height);
              }
              rctx.factory->end_render_pass(ctx);
              //}
              //fprintf(stdout, "[END RECORDING] thread %i\n", thread_id);
              //fflush(stdout);
              if (--new_work == 0) {
                work_finish_cv.notify_one();
              }
            }
          }));
          // timestamps.insert(rctx.factory, ctx);
        }
      }
      ito(0x100) one_time_wake[i] = true;
      new_work                    = threads.size();
      work_start_cv.notify_all();
      std::unique_lock<std::mutex> lk(work_finish_mutex);
      work_finish_cv.wait(lk, [&] { return new_work == 0; });
      new_work = 0;
      rctx.gizmo_layer->reset();
      //fprintf(stdout, "[END FRAME]\n");
      //fflush(stdout);
      // for (auto &th : threads) th.join();
      // threads.clear();
    }
  }
  void release(rd::IDevice *factory) { factory->release_resource(normal_rt); }
};

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
    // bool show = true;
    // ShowExampleAppCustomNodeGraph(&show);
    // ImGui::TestNodeGraphEditor();
    // ImGui::Begin("Text");
    // te.Render("Editor");
    // ImGui::End();
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
    ImGui::Text("%fms", gbuffer_pass.timestamps.duration);
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
    rctx.gizmo_layer = Gizmo_Layer::create(factory);
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

    gbuffer_pass.init();
  }
  void on_release() override { //
    // thread_pool.release();
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
    rctx.scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GizmoComponent>() == NULL) {
          GizmoComponent::create(rctx.gizmo_layer, mn);
        }
      }
    });
    rctx.scene->get_root()->update();
    gbuffer_pass.render(rctx);
  }
};
#endif
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  static int init = [] { return SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS); }();

  auto window_loop = [](rd::Impl_t impl) {
    SDL_Window *window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          512, 512, SDL_WINDOW_RESIZABLE);
    void *      handle = NULL;
#ifdef WIN32
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;
    handle    = (void *)hwnd;
#else
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    Window            hwnd       = wmInfo.info.x11.window;
    xcb_connection_t *connection = XGetXCBConnection(wmInfo.info.x11.display);
    Ptr2              ptrs{(void *)hwnd, (void *)connection};
    handle = (void *)&ptrs;
#endif
    rd::IDevice *factory = NULL;
    if (impl == rd::Impl_t::VULKAN) {
      factory = rd::create_vulkan(handle);
    } else if (impl == rd::Impl_t::DX12) {
      factory = rd::create_dx12(handle);
    } else {
      TRAP;
    }
    if (factory == NULL) return;
    factory->start_frame();
    factory->end_frame();

    constexpr u32 MAX_FRAMES = 0x10;
    u32           NUM_FRAMES = factory->get_num_swapchain_images();
    Resource_ID   render_passes[MAX_FRAMES]{};
    Resource_ID   psos[MAX_FRAMES]{};
    defer(ito(NUM_FRAMES) {
      if (render_passes[i].is_valid()) factory->release_resource(render_passes[i]);
      if (psos[i].is_valid()) factory->release_resource(psos[i]);
    });
    u32  frame_id = 0;
    u32  width = 0, height = 0;
    u32  window_id    = SDL_GetWindowID(window);
    bool focus_events = false;
    struct PushConstants {
      float angle;
    };
    Resource_ID signature = [=] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return factory->create_signature(table_info);
    }();
    static string_ref            shader    = stref_s(R"(

struct PushConstants
{
  float angle;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos   : SV_POSITION;
};

#ifdef VERTEX

PSInput main(in uint id : SV_VertexID) {
  PSInput output;
  static float2 pos_arr[3] = {
    float2(-1.0f, -1.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, -1.0f),
  };
  float2x2 rot = {
    cos(pc.angle), sin(pc.angle),
    -sin(pc.angle), cos(pc.angle)
  };
  output.pos   = float4(mul(rot, pos_arr[id]), 0.0f, 1.0f);
  return output;
}
#endif
#ifdef PIXEL

float4 main(in PSInput input) : SV_TARGET0 {
  return float4(1.0f, 0.0f, 0.0f, 1.0f);
}
#endif

)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    Resource_ID vs = factory->create_shader(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    Resource_ID ps = factory->create_shader(rd::Stage_t::PIXEL, shader, &defines[1], 1);
    defer({
      factory->release_resource(vs);
      factory->release_resource(ps);
    });
    rd::IBinding_Table *table = factory->create_binding_table(signature);
    while (true) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (focus_events) {
          if (event.type == SDL_QUIT) {
            return;
          }
        }
        if (event.type == SDL_WINDOWEVENT) {
          if (event.window.windowID != window_id) continue;
          if (event.window.event == SDL_WINDOWEVENT_ENTER) {
            focus_events = true;
          } else if (event.window.event == SDL_WINDOWEVENT_LEAVE) {
            focus_events = false;
          } else if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
          }
        }
      }
      factory->start_frame();
      rd::Image2D_Info scinfo = factory->get_swapchain_image_info();
      if (width != scinfo.width || height != scinfo.height) {
        width  = scinfo.width;
        height = scinfo.height;
        ito(NUM_FRAMES) {
          if (render_passes[i].is_valid()) factory->release_resource(render_passes[i]);
          if (psos[i].is_valid()) factory->release_resource(psos[i]);
          render_passes[i] = {};
          psos[i]          = {};
        }
      }
      if (render_passes[frame_id].is_null()) {
        rd::Render_Pass_Create_Info info;
        MEMZERO(info);
        rd::RT_View rt0;
        MEMZERO(rt0);
        rt0.image             = factory->get_swapchain_image();
        rt0.format            = rd::Format::NATIVE;
        rt0.clear_color.clear = true;
        rt0.clear_color.g     = 1.0f;
        info.rts.push(rt0);
        render_passes[frame_id] = factory->create_render_pass(info);
        rd::Graphics_Pipeline_State gfx_state{};
        setup_default_state(gfx_state);
        rd::Blend_State bs;
        MEMZERO(bs);
        bs.enabled        = true;
        bs.alpha_blend_op = rd::Blend_OP::ADD;
        bs.color_blend_op = rd::Blend_OP::ADD;
        bs.dst_alpha      = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
        bs.src_alpha      = rd::Blend_Factor::SRC_ALPHA;
        bs.dst_color      = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
        bs.src_color      = rd::Blend_Factor::SRC_ALPHA;
        bs.color_write_mask =
            (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
            (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
        gfx_state.OM_set_blend_state(0, bs);
        gfx_state.VS_set_shader(vs);
        gfx_state.PS_set_shader(ps);
        /* {
           rd::Attribute_Info info;
           MEMZERO(info);
           info.binding  = 0;
           info.format   = rd::Format::RG32_FLOAT;
           info.location = 0;
           info.offset   = 0;
           info.type     = rd::Attriute_t::POSITION;
           gfx_state.IA_set_attribute(info);
         }
         {
           rd::Attribute_Info info;
           MEMZERO(info);
           info.binding  = 0;
           info.format   = rd::Format::RG32_FLOAT;
           info.location = 1;
           info.offset   = 8;
           info.type     = rd::Attriute_t::TEXCOORD0;
           gfx_state.IA_set_attribute(info);
         }
         {
           rd::Attribute_Info info;
           MEMZERO(info);
           info.binding  = 0;
           info.format   = rd::Format::RGBA8_UNORM;
           info.location = 2;
           info.offset   = 16;
           info.type     = rd::Attriute_t::TEXCOORD1;
           gfx_state.IA_set_attribute(info);
         }*/
        psos[frame_id] =
            factory->create_graphics_pso(signature, render_passes[frame_id], gfx_state);
      }
      rd::ICtx *ctx = factory->start_render_pass(render_passes[frame_id]);
      ctx->start_render_pass();
      auto         sc_info = factory->get_swapchain_image_info();
      static float angle   = 0.0f;
      angle += 1.0e-3f;
      table->push_constants(&angle, 0, sizeof(angle));
      ctx->bind_table(table);
      ctx->set_scissor(0, 0, sc_info.width, sc_info.height);
      ctx->set_viewport(0.0f, 0.0f, (f32)sc_info.width, (f32)sc_info.height, 0.0f, 1.0f);
      ctx->bind_graphics_pso(psos[frame_id]);
      ctx->draw(3, 1, 0, 0);
      ctx->end_render_pass();
      factory->end_render_pass(ctx);
      factory->end_frame();
      frame_id = (frame_id + 1) % NUM_FRAMES;
    }
  };
  std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  std::thread dx12_thread   = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  vulkan_thread.join();
  dx12_thread.join();
  // IGUIApp::start<Event_Consumer>(rd::Impl_t::VULKAN);
  return 0;
}
