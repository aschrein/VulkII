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
  rd::IDevice *factory = NULL;
  Config *     config  = NULL;
  Scene *      scene   = NULL;
  // Gizmo_Layer * gizmo_layer = NULL;
};
class GBufferPass {
  public:
  Resource_ID signature{};
  Resource_ID pso{};
  Resource_ID pass{};
  Resource_ID normal_rt{};
  Resource_ID depth_rt{};
  Resource_ID gbuffer_vs{};
  Resource_ID gbuffer_ps{};
  u32         width  = 0;
  u32         height = 0;

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

  public:
  // TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
    gbuffer_vs            = rctx.factory->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::binding(0, 0)]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

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
    gbuffer_ps            = rctx.factory->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
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
    Resource_ID signature = [=] {
      rd::Binding_Space_Create_Info set_info{};
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return rctx.factory->create_signature(table_info);
    }();
  }
  void remake_pass(RenderingContext rctx) {
    if (pass.is_valid()) rctx.factory->release_resource(pass);
    if (pso.is_valid()) rctx.factory->release_resource(pso);
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
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      rd::RT_View                 rt0{};
      rt0.image             = normal_rt;
      rt0.format            = rd::Format::NATIVE;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      info.depth_target.image             = depth_rt;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::NATIVE;
      return rctx.factory->create_render_pass(info);
    }();
    pso = [=] {
      rd::Graphics_Pipeline_State gfx_state{};
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state{};
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      gfx_state.DS_set_state(ds_state);
      rd::Blend_State bs{};
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
      gfx_state.VS_set_shader(gbuffer_vs);
      gfx_state.PS_set_shader(gbuffer_ps);
      {
        rd::Attribute_Info info{};
        info.binding  = 0;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info{};
        info.binding  = 0;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 1;
        info.offset   = 8;
        info.type     = rd::Attriute_t::TEXCOORD0;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info{};
        info.binding  = 0;
        info.format   = rd::Format::RGBA8_UNORM;
        info.location = 2;
        info.offset   = 16;
        info.type     = rd::Attriute_t::TEXCOORD1;
        gfx_state.IA_set_attribute(info);
      }
      gfx_state.IA_set_vertex_binding(0, sizeof(ImDrawVert), rd::Input_Rate::VERTEX);
      return rctx.factory->create_graphics_pso(signature, pass, gfx_state);
    }();
  }
  void render(RenderingContext rctx) {
    // timestamps.update(rctx.factory);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f,
    // 0.0f));
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
    // fprintf(stdout, "[START FRAME]\n");
    // fflush(stdout);

    ito(2) {
      int _thread_id = i;
      {
        int thread_id = _thread_id;
        u32 width     = rctx.config->get_u32("g_buffer_width");
        u32 height    = rctx.config->get_u32("g_buffer_height");

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
        ctx->push_constants(&viewproj, 0, sizeof(float4x4));
        setup_default_state(ctx, 1);

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

        rctx.gizmo_layer->render(rctx.factory, ctx, width, height);

        rctx.factory->end_render_pass(ctx);
      }
      // timestamps.insert(rctx.factory, ctx);
    }

    // rctx.gizmo_layer->reset();
    // fprintf(stdout, "[END FRAME]\n");
    // fflush(stdout);
    // for (auto &th : threads) th.join();
    // threads.clear();
  }
} void release(rd::IDevice *factory) {
  factory->release_resource(normal_rt);
}
}
;
#endif
#if 1
class Event_Consumer : public IGUIApp {
  public:
  // GBufferPass      gbuffer_pass;
  // RenderingContext rctx{};
  void init_traverse(List *l) {
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
    // ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
    // ImVec2(wsize.x, wsize.y));
    //{ Ray ray = rctx.gizmo_layer->getMouseRay(); }
    ImGui::End();
  }
  void on_init() override { //
    // rctx.factory = this->factory;
    TMP_STORAGE_SCOPE;
    // rctx.gizmo_layer = Gizmo_Layer::create(factory);
    // new XYZDragGizmo(gizmo_layer, &pos);
    /* rctx.scene  = Scene::create();
     rctx.config = new Config;
     rctx.config->init(stref_s(R"(
 (
  (add u32  g_buffer_width 512 (min 4) (max 1024))
  (add u32  g_buffer_height 512 (min 4) (max 1024))
 )
 )"));
     rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
     rctx.scene->update();*/
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }

    // gbuffer_pass.init();
  }
  void on_release() override { //
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
    });
    rctx.scene->get_root()->update();
    gbuffer_pass.render(rctx);*/
  }
};
#endif
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
#if 0
  static int init = [] { return SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS); }();

  auto window_loop = [](rd::Impl_t impl) {
    SDL_Window *window = NULL;
    {
      static std::mutex           mutex;
      std::lock_guard<std::mutex> _locke(mutex);
      window = SDL_CreateWindow(impl == rd::Impl_t::VULKAN ? "VulkII on Vulkan" : "VulkII on DX12",
                                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 512, 512,
                                SDL_WINDOW_RESIZABLE);
    }
    void *handle = NULL;
#  ifdef WIN32
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;
    handle    = (void *)hwnd;
#  else
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    Window            hwnd       = wmInfo.info.x11.window;
    xcb_connection_t *connection = XGetXCBConnection(wmInfo.info.x11.display);
    Ptr2              ptrs{(void *)hwnd, (void *)connection};
    handle = (void *)&ptrs;
#  endif
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
    // struct PushConstants {
    //  float angle;
    //};

    static string_ref shader = stref_s(R"(

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
    struct PushConstants {
      float2 uScale;
      float2 uTranslate;
      u32    control_flags;
    };
    static string_ref imgui_shader = stref_s(R"(
struct PushConstants
{
  float2 uScale;
  float2 uTranslate; 
  u32    control_flags;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

#define CONTROL_DEPTH_ENABLE 1
#define CONTROL_AMPLIFY 2
#define is_control(flag) (pc.control_flags & flag) != 0

[[vk::binding(0, 0)]] Texture2D<float4>   sTexture          : register(t0, space0);
[[vk::binding(1, 0)]] SamplerState        sSampler          : register(s1, space0);

struct PSInput {
  [[vk::location(0)]] float4 pos   : SV_POSITION;
  [[vk::location(1)]] float2 UV    : TEXCOORD0;
  [[vk::location(2)]] float4 Color : TEXCOORD1;
};

#ifdef VERTEX

struct VSInput {
  [[vk::location(0)]] float2 aPos    : POSITION;
  [[vk::location(1)]] float2 aUV     : TEXCOORD0;
  [[vk::location(2)]] float4 aColor  : TEXCOORD1;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.Color = input.aColor;
  output.UV    = input.aUV;
  output.pos   = float4(input.aPos * pc.uScale + pc.uTranslate, 0.0f, 1.0f);
  output.pos.y *= -1.0f;
  return output;
}
#endif
#ifdef PIXEL

float4 main(in PSInput input) : SV_TARGET0 {
  if (is_control(CONTROL_DEPTH_ENABLE)) {
    float depth = sTexture.Sample(sSampler, input.UV).r;
    depth = pow(depth * 500.0f, 1.0f / 2.0f);
    return float4_splat(depth);
  } else {
    float4 color = input.Color * sTexture.Sample(sSampler, input.UV);
    if (is_control(CONTROL_AMPLIFY)) {
      color *= 10.0f;
      color.a = 1.0f;
    }
    return color;
  }
}
#endif
)");
    Resource_ID       signature    = [=] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return factory->create_signature(table_info);
    }();
    defer(factory->release_resource(signature));
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    Resource_ID vs = factory->create_shader(rd::Stage_t::VERTEX, imgui_shader, &defines[0], 1);
    Resource_ID ps = factory->create_shader(rd::Stage_t::PIXEL, imgui_shader, &defines[1], 1);
    defer({
      factory->release_resource(vs);
      factory->release_resource(ps);
    });

    Resource_ID sampler = [=] {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
      return factory->create_sampler(info);
    }();
    IMGUI_CHECKVERSION();
    ImGuiContext *imgui_ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(imgui_ctx);
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    {
      static std::mutex           mutex;
      std::lock_guard<std::mutex> _locke(mutex);
      if (factory->getImplType() == rd::Impl_t::VULKAN) {
        ImGui_ImplSDL2_InitForVulkan(window);
      } else if (factory->getImplType() == rd::Impl_t::DX12) {
        ImGui_ImplSDL2_InitForD3D(window);
      } else {
        TRAP;
      }
    }
    Resource_ID font_texture = [&] {
      static std::mutex           mutex;
      std::lock_guard<std::mutex> _locke(mutex);
      unsigned char *             font_pixels = NULL;
      int                         font_width = 0, font_height = 0;
      io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_width, &font_height);

      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format = rd::Format::RGBA8_UNORM;
      info.width  = font_width;
      info.height = font_height;
      info.depth  = 1;
      info.layers = 1;
      info.levels = 1;
      info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
      Resource_ID font_texture = factory->create_image(info);
      io.Fonts->TexID          = (ImTextureID)font_texture.data;

      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      u32 pitch = rd::IDevice::align_up(font_width * 4, rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);
      buf_info.size              = font_height * pitch;
      Resource_ID staging_buffer = factory->create_buffer(buf_info);
      factory->release_resource(staging_buffer);

      u8 *dst = (u8 *)factory->map_buffer(staging_buffer);
      yto(font_height) {
        xto(font_width) {
          memcpy(dst + pitch * y, font_pixels + font_width * 4 * y, font_width * 4);
        }
      }
      factory->unmap_buffer(staging_buffer);
      rd::ICtx *ctx = factory->start_compute_pass();
      ctx->image_barrier(font_texture, rd::Image_Access::TRANSFER_DST);
      ctx->copy_buffer_to_image(staging_buffer, 0, font_texture, rd::Image_Copy::top_level(pitch));
      factory->end_compute_pass(ctx);
      font_pixels = NULL;
      return font_texture;
    }();
    // static InlineArray<SDL_Event, 0x100> sdl_event_queue;
    // static std::mutex                    sdl_event_mutex;
    // static std::atomic<int>              sdl_event_cnt;
    while (true) {
      ImGui::SetCurrentContext(imgui_ctx);
      SDL_Event event{};
      {
        // std::lock_guard<std::mutex> _locke(sdl_event_mutex);
        while (SDL_PollEvent(&event)) {
          // SDL_Event event = sdl_event_queue[i];
          if (focus_events) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
            }
          }
          if (event.type == SDL_WINDOWEVENT) {
            if (event.window.windowID != window_id) continue;
            if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
              return;
            } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
                       event.window.event == SDL_WINDOWEVENT_ENTER) {
              focus_events = true;
            } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
                       event.window.event == SDL_WINDOWEVENT_LEAVE) {
              focus_events = false;
            } else if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
            }
          }
          // sdl_event_queue.push(event);
          // sdl_event_cnt = 2;
        }
        /* if (int val = sdl_event_cnt.fetch_add(-1) > 0) {
           defer({
             if (val == 1) {
               sdl_event_queue.size = 0;
             }
           });
           ito(sdl_event_queue.size) {
             SDL_Event event = sdl_event_queue[i];
             if (focus_events) {
               ImGui_ImplSDL2_ProcessEvent(&event);
               if (event.type == SDL_QUIT) {
               }
             }
             if (event.type == SDL_WINDOWEVENT) {
               if (event.window.windowID != window_id) continue;
               if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                 return;
               } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
                          event.window.event == SDL_WINDOWEVENT_ENTER) {
                 focus_events = true;
               } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
                          event.window.event == SDL_WINDOWEVENT_LEAVE) {
                 focus_events = false;
               } else if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
               }
             }
           }
         }*/
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
        {
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
        }
        gfx_state.IA_set_vertex_binding(0, sizeof(ImDrawVert), rd::Input_Rate::VERTEX);
        psos[frame_id] =
            factory->create_graphics_pso(signature, render_passes[frame_id], gfx_state);
      }
      rd::ICtx *ctx = factory->start_render_pass(render_passes[frame_id]);
      ctx->image_barrier(font_texture, rd::Image_Access::SAMPLED);
      ctx->start_render_pass();
      auto         sc_info = factory->get_swapchain_image_info();
      static float angle   = 0.0f;
      angle += 1.0e-3f;
      // ctx->set_scissor(0, 0, sc_info.width, sc_info.height);
      ctx->set_viewport(0.0f, 0.0f, (f32)sc_info.width, (f32)sc_info.height, 0.0f, 1.0f);
      // ImGui Pass
      {
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
        ImGuiViewport *  viewport     = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        window_flags |= ImGuiWindowFlags_NoBackground;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 1.0f);
        ImGui::SetNextWindowBgAlpha(-1.0f);
        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(4);
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::End();

        static bool show_demo_window = true;
        ImGui::ShowDemoWindow(&show_demo_window);

        ImGui::Render();
      }
      ImDrawData *draw_data = ImGui::GetDrawData();
      // Render IMGUI
      {
        Resource_ID vertex_buffer = [=] {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
          buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
          buf_info.size        = (draw_data->TotalVtxCount + 1) * sizeof(ImDrawVert);
          return factory->create_buffer(buf_info);
        }();
        factory->release_resource(vertex_buffer);
        Resource_ID index_buffer = [=] {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
          buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
          buf_info.size        = (draw_data->TotalIdxCount + 1) * sizeof(ImDrawIdx);
          return factory->create_buffer(buf_info);
        }();
        factory->release_resource(index_buffer);
        {
          {
            ImDrawVert *vtx_dst = (ImDrawVert *)factory->map_buffer(vertex_buffer);

            ito(draw_data->CmdListsCount) {

              const ImDrawList *cmd_list = draw_data->CmdLists[i];
              memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
                     cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));

              vtx_dst += cmd_list->VtxBuffer.Size;
            }
            factory->unmap_buffer(vertex_buffer);
          }
          {
            ImDrawIdx *idx_dst = (ImDrawIdx *)factory->map_buffer(index_buffer);
            ito(draw_data->CmdListsCount) {

              const ImDrawList *cmd_list = draw_data->CmdLists[i];

              memcpy(idx_dst, cmd_list->IdxBuffer.Data,
                     cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
              idx_dst += cmd_list->IdxBuffer.Size;
            }
            factory->unmap_buffer(index_buffer);
          }
        }
        ImVec2 clip_off          = draw_data->DisplayPos;
        ImVec2 clip_scale        = draw_data->FramebufferScale;
        int    global_vtx_offset = 0;
        int    global_idx_offset = 0;

        ctx->bind_graphics_pso(psos[frame_id]);
        if (sizeof(ImDrawIdx) == 2)
          ctx->bind_index_buffer(index_buffer, 0, rd::Index_t::UINT16);
        else
          ctx->bind_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
        ctx->bind_vertex_buffer(0, vertex_buffer, 0);

        u32 control = 0;
        for (int n = 0; n < draw_data->CmdListsCount; n++) {
          const ImDrawList *cmd_list = draw_data->CmdLists[n];
          for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
            const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[cmd_i];
            ImVec4           clip_rect;
            clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
            clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
            clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
            clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;
            // ImGui_ID img = image_bindings[(size_t)pcmd->TextureId];
            /*if (img.format == rd::Format::D32_FLOAT) {
              control = 1;
            }
            if (img.format == rd::Format::R32_UINT) {
              control    = 2;
              img.format = rd::Format::RGBA8_UNORM;
            }
            if (control != 0) {
              ctx->push_constants(&control, 16, 4);
            }*/

            rd::IBinding_Table *table = factory->create_binding_table(signature);
            defer(table->release());
            {
              float scale[2];
              scale[0] = 2.0f / draw_data->DisplaySize.x;
              scale[1] = 2.0f / draw_data->DisplaySize.y;
              float translate[2];
              translate[0] = -1.0f - draw_data->DisplayPos.x * scale[0];
              translate[1] = -1.0f - draw_data->DisplayPos.y * scale[1];
              table->push_constants(scale, 0, 8);
              table->push_constants(translate, 8, 8);
              table->push_constants(&control, 16, 4);
            }
            /*rd::Image_Subresource range;
            range.layer      = img.base_layer;
            range.level      = img.base_level;
            range.num_layers = 1;
            range.num_levels = 1;*/
            table->bind_sampler(0, 1, sampler);
            Resource_ID img_id{};
            img_id.data = (u64)pcmd->TextureId;
            table->bind_texture(0, 0, 0, img_id, rd::Image_Subresource::all_levels(),
                                rd::Format::NATIVE);
            ctx->bind_table(table);

            ctx->set_scissor(clip_rect.x, clip_rect.y, clip_rect.z - clip_rect.x,
                             clip_rect.w - clip_rect.y);
            ctx->draw_indexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, 0,
                              pcmd->VtxOffset + global_vtx_offset);
          }
          global_idx_offset += cmd_list->IdxBuffer.Size;
          global_vtx_offset += cmd_list->VtxBuffer.Size;
        }
      }
      //
      // ctx->draw(3, 1, 0, 0);
      ctx->end_render_pass();
      factory->end_render_pass(ctx);
      factory->end_frame();
      frame_id = (frame_id + 1) % NUM_FRAMES;
    }
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext(imgui_ctx);
  };
  // std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  std::thread dx12_thread = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  // vulkan_thread.join();
  dx12_thread.join();
#endif
  IGUIApp::start<Event_Consumer>(rd::Impl_t::VULKAN);
  return 0;
}
