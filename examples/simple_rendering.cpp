#include "rendering.hpp"
#include "script.hpp"

#include "rendering_utils.hpp"
#include "scene.hpp"

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

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

Image2D_Raw const *g_image     = NULL;
Resource_ID        g_image_gfx = {};

static int g_init = []() {
  TMP_STORAGE_SCOPE;
  g_camera.init();
  g_config.init(stref_s(R"(
(
  (add u32 displayed_mip 0 (min 0) (max 10))
)
)"));

  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
    init_traverse(cur);
  }
  g_scene.init();
  u32 id =
      g_scene.load_image(stref_s("images/9194.jpg"), rd::Format::RGBA8_SRGBA);
  g_image = &g_scene.get_image(id);
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

Mip_Builder g_mip_builder;

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

class Init_Pass : public rd::IPass {
  bool mip_built;

  public:
  Init_Pass() { mip_built = false; }
  void on_end(rd::IResource_Manager *rm) override {}
  void on_begin(rd::IResource_Manager *pc) override { g_mip_builder.init(pc); }
  void exec(rd::Imm_Ctx *ctx) override {
    if (mip_built == false && g_image_gfx.is_null() == false) {
      mip_built = true;
      g_mip_builder.compute(ctx, *g_image, g_image_gfx);
    }
  }
  string_ref get_name() override { return stref_s("Init_Pass"); }
  void       release(rd::IResource_Manager *rm) override {
    g_mip_builder.release(rm);
  }
};

class GUI_Pass : public IGUI_Pass {
  public:
  GUI_Pass() {}
  void exec(rd::Imm_Ctx *ctx) override { IGUI_Pass::exec(ctx); }
  void on_gui(rd::IResource_Manager *pc) override {
    {
      ImGui::Begin("Config");
      g_config.on_imgui();
      ImGui::End();
    }
    {
      ImGui::Begin("pass");
      {
        auto wsize = get_window_size();
        if (g_image_gfx.is_null()) {
          g_image_gfx = Mip_Builder::create_image(pc, *g_image, true);
        } else {
          ImGui::Image(bind_texture(g_image_gfx, 0, g_config.get_u32("displayed_mip"),
                                    rd::Format::RGBA8_UNORM),
                       ImVec2(wsize.x, wsize.y));
        }
        auto wpos = ImGui::GetCursorScreenPos();

        ImGuiIO &io = ImGui::GetIO();
        // g_config.get_u32("g_buffer_width")  = wsize.x;
        // g_config.get_u32("g_buffer_height") = wsize.y;
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
        ImGui::End();
      }
    }
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  GUI_Pass *    gui  = new GUI_Pass;
  MEMZERO(g_mip_builder);
  pmng->set_event_consumer(gui);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Init_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, gui);
  pmng->loop();
  return 0;
}
