#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <imgui.h>

Config g_config;
Camera g_camera;
Scene *g_scene = Scene::create();

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
(
 (add u32  g_buffer_width 512 (min 4) (max 1024))
 (add u32  g_buffer_height 512 (min 4) (max 1024))
 (add bool forward 1)
 (add bool "depth test" 1)
 (add f32  strand_size 1.0 (min 0.1) (max 16.0))
)
)"));

  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
    init_traverse(cur);
  }

  g_scene->load_mesh(stref_s("mesh"), stref_s("models/low_poly_ellie/scene.gltf"));

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

class GizmoPass {
  public:
  Gizmo_Layer gizmo_layer;

  public:
  void init(rd::IFactory *factory) { gizmo_layer.init(factory); }
  void release(rd::IFactory *factory) { gizmo_layer.release(factory); }
  void render(rd::IFactory *factory, rd::Imm_Ctx *ctx) {
    {
      float dx = 1.0e-1f * g_camera.distance;
      gizmo_layer.draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
      gizmo_layer.draw_cylinder(g_camera.look_at, g_camera.look_at + float3{dx, 0.0f, 0.0f},
                                dx * 0.04f, float3{1.0f, 0.0f, 0.0f});
      gizmo_layer.draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, dx, 0.0f},
                                dx * 0.04f, float3{0.0f, 1.0f, 0.0f});
      gizmo_layer.draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, 0.0f, dx},
                                dx * 0.04f, float3{0.0f, 0.0f, 1.0f});
    }
    gizmo_layer.render(factory, ctx, g_camera.viewproj());
  }
};

template <typename T> class GPUBuffer {
  private:
  rd::IFactory *factory;
  Array<T>      cpu_array;
  Resource_ID   gpu_buffer;
  Resource_ID   cpu_buffer;
  size_t        gpu_buffer_size;

  public:
  void init(rd::IFactory *factory) {
    this->factory = factory;
    cpu_array.init();
    gpu_buffer.reset();
    cpu_buffer.reset();
    gpu_buffer_size = 0;
  }
  void push(T a) { cpu_array.push(a); }
  void clear() {
    cpu_array.release();
    factory->release_resource(gpu_buffer);
    factory->release_resource(cpu_buffer);
    gpu_buffer.reset();
  }
  void reset() { cpu_array.reset(); }
  void flush() {
    if (gpu_buffer.is_null()) {
      if (cpu_buffer) factory->release_resource(cpu_buffer);

    }
  }
  void release() {
    factory->release_resource(gpu_buffer);
    factory->release_resource(cpu_buffer);
    cpu_array.release();
  }
};

class GBufferPass {
  public:
  Resource_ID normal_rt;
  Resource_ID depth_rt;

  public:
  void init() { MEMZERO(*this); }
  void render(rd::IFactory *factory) {
    u32 width  = g_config.get_u32("g_buffer_width");
    u32 height = g_config.get_u32("g_buffer_height");
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
      normal_rt = get_or_create_image(factory, rt0_info, normal_rt);
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
      depth_rt            = get_or_create_image(factory, rt0_info, depth_rt);
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
      info.rts.push(rt0);

      info.depth_target.image             = depth_rt;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::NATIVE;

      rd::Imm_Ctx *ctx = factory->start_render_pass(info);
      ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float4x4)  (name world_transform))
)

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))
@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_OUTPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(ENTRY)
  PIXEL_POSITION   = POSITION;
  PIXEL_NORMAL     = NORMAL;
  PIXEL_TEXCOORD0  = TEXCOORD0;
  @(EXPORT_POSITION mul4(viewproj, float4(POSITION, 1.0)));
@(END)
)"),
                                                    NULL, 0));
      ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  float4 color = float4(PIXEL_NORMAL, 1.0);
  @(EXPORT_COLOR 0 color);
@(END)
)"),
                                                    NULL, 0));
      static u32 attribute_to_location[] = {
          0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
      };
      setup_default_state(ctx, 1);
      rd::DS_State ds_state;
      MEMZERO(ds_state);
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      ctx->DS_set_state(ds_state);
      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      ctx->RS_set_state(rs_state);
      float4x4 viewproj = g_camera.viewproj();
      ctx->push_constants(&viewproj, 0, sizeof(float4x4));
      g_scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (mn->getComponent<GfxSufraceComponent>() == NULL) {
            GfxSufraceComponent::create(factory, mn);
          }
          GfxSufraceComponent *gs = mn->getComponent<GfxSufraceComponent>();
          ito(gs->getNumSurfaces()) {
            GfxSurface *s = gs->getSurface(i);
            s->draw(ctx, attribute_to_location);
          }
        }
      });
      factory->end_render_pass(ctx);
    }
  }
  void release(rd::IFactory *factory) { factory->release_resource(normal_rt); }
};

class Event_Consumer : public IGUI_Pass {
  Resource_ID buf_id = {};
  // TextEditor  te;
  GBufferPass gbuffer_pass;

  public:
  void init(rd::Pass_Mng *pmng) override { //
    IGUI_Pass::init(pmng);
    g_camera.init();
    gbuffer_pass.init();
    /* te.SetText(R"(
 (
  (add u32  g_buffer_width 512 (min 4) (max 1024))
  (add u32  g_buffer_height 512 (min 4) (max 1024))
  (add bool forward 1)
  (add bool "depth test" 1)
  (add f32  strand_size 1.0 (min 0.1) (max 16.0))
 )
 )");*/
  }
  void on_gui(rd::IFactory *factory) override { //
    // bool show = true;
    // ShowExampleAppCustomNodeGraph(&show);
    // ImGui::TestNodeGraphEditor();
    // ImGui::Begin("Text");
    // te.Render("Editor");
    // ImGui::End();
    ImGui::Begin("Config");
    g_config.on_imgui();
    // ImGui::LabelText("clear pass", "%f ms", hr.clear_timestamp.duration);
    // ImGui::LabelText("pre  pass", "%f ms", hr.prepass_timestamp.duration);
    // ImGui::LabelText("resolve pass", "%f ms", hr.resolve_timestamp.duration);
    ImGui::End();
    ImGui::Begin("main viewport");

    auto wsize = get_window_size();
    ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                 ImVec2(wsize.x, wsize.y));
    auto wpos = ImGui::GetCursorScreenPos();
    // auto iinfo      = factory->get_image_info(hr.hair_img);
    // g_camera.aspect = float(iinfo.height) / iinfo.width;
    ImGuiIO &io = ImGui::GetIO();
    if (ImGui::IsWindowHovered()) {
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
      ImVec2 mpos    = ImGui::GetMousePos();
      i32    cur_m_x = mpos.x;
      i32    cur_m_y = mpos.y;
      if (io.MouseDown[0] && last_m_x > 0) {
        i32 dx = cur_m_x - last_m_x;
        i32 dy = cur_m_y - last_m_y;
        g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
        g_camera.theta -= (float)(dy)*5.0e-3f;
      }
      last_m_x = cur_m_x;
      last_m_y = cur_m_y;
    }
    g_camera.update();
    ImGui::End();
  }
  void on_init(rd::IFactory *factory) override { //
  }
  void on_release(rd::IFactory *factory) override { //
    g_scene->release();
    IGUI_Pass::release(factory);
  }
  void consume(void *_event) override { //
    IGUI_Pass::consume(_event);
  }
  void on_frame(rd::IFactory *factory) override { //
    gbuffer_pass.render(factory);
    IGUI_Pass::on_frame(factory);
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  IGUI_Pass *   gui  = new Event_Consumer;
  gui->init(pmng);
  pmng->set_event_consumer(gui);
  pmng->loop();
  return 0;
}
