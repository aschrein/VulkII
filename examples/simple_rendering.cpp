
#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <atomic>
//#include <functional>
#include <3rdparty/half.hpp>
#include <imgui.h>
#include <mutex>
#include <thread>

struct RenderingContext {
  rd::IFactory *factory     = NULL;
  Config *      config      = NULL;
  Scene *       scene       = NULL;
  Gizmo_Layer * gizmo_layer = NULL;
};
class GBufferPass {
  public:
  Resource_ID normal_rt;
  Resource_ID depth_rt;

  struct GPU_Cube {
    Resource_ID vertex_buffer = {};
    Resource_ID index_buffer  = {};
    void        init(rd::IFactory *factory) {
      if (vertex_buffer.is_null()) {

        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER |
                              (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        buf_info.size = sizeof(float3) * 8;
        vertex_buffer = factory->create_buffer(buf_info);

        float3 dvertices[8] = {float3(-1, -1, -1), float3(1, -1, -1), float3(1, 1, -1),
                               float3(-1, 1, -1),  float3(-1, -1, 1), float3(1, -1, 1),
                               float3(1, 1, 1),    float3(-1, 1, 1)};
        init_buffer(factory, vertex_buffer, dvertices, sizeof(dvertices));
      }
      if (index_buffer.is_null()) {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER |
                              (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        buf_info.size           = 6 * 2 * 3 * sizeof(u32);
        index_buffer            = factory->create_buffer(buf_info);
        u32 dindices[6 * 2 * 3] = {0, 1, 3, 3, 1, 2, 1, 5, 2, 2, 5, 6, 5, 4, 6, 6, 4, 7,
                                   4, 0, 7, 7, 0, 3, 3, 2, 7, 7, 2, 6, 4, 5, 0, 0, 5, 1};
        init_buffer(factory, index_buffer, dindices, sizeof(dindices));
      }
    }
    void bind(rd::Imm_Ctx *ctx) {
      ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, 12, rd::Input_Rate::VERTEX);
      ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        ctx->IA_set_attribute(info);
      }
    }
    void draw(rd::Imm_Ctx *ctx) { ctx->draw_indexed(36, 1, 0, 0, 0); }
    void release(rd::IFactory *factory) {
      if (vertex_buffer.is_null() == false) factory->release_resource(vertex_buffer);
      if (index_buffer.is_null() == false) factory->release_resource(index_buffer);
    }
  };

  public:
  TimeStamp_Pool timestamps = {};

  void init() { MEMZERO(*this); }
  void render(RenderingContext rctx) {
    timestamps.update(rctx.factory);
    float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f, 0.0f));
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
    {
      rd::Render_Pass_Create_Info info;
      MEMZERO(info);
      info.width  = width;
      info.height = height;
      rd::RT_View rt0;
      MEMZERO(rt0);
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

      rd::Imm_Ctx *ctx = rctx.factory->start_render_pass(info);
      timestamps.insert(rctx.factory, ctx);
      setup_default_state(ctx, 1);
      rd::DS_State ds_state;
      rd::RS_State rs_state;
      float4x4     viewproj = rctx.gizmo_layer->get_camera().viewproj();
      ctx->push_constants(&viewproj, 0, sizeof(float4x4));
      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      MEMZERO(ds_state);
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      ctx->DS_set_state(ds_state);

      ctx->VS_set_shader(rctx.factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

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
                                                         NULL, 0));
      ctx->PS_set_shader(rctx.factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
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
                                                         NULL, 0));
      static u32 attribute_to_location[] = {
          0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
      };
      if (rctx.config->get_bool("render_scene")) {
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
      }
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::LINE;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      ctx->RS_set_state(rs_state);
      ctx->RS_set_depth_bias(0.1f);
      if (rctx.config->get_bool("render_scene_wireframe")) {
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
      }
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
        rctx.gizmo_layer->render(rctx.factory, ctx, width, height);
      }
      rctx.gizmo_layer->reset();
      timestamps.insert(rctx.factory, ctx);
      rctx.factory->end_render_pass(ctx);
    }
  }
  void release(rd::IFactory *factory) { factory->release_resource(normal_rt); }
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

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  IGUIApp::start<Event_Consumer>(rd::Impl_t::VULKAN);
  return 0;
}
