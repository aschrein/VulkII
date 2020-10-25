#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <atomic>
//#include <functional>
#include <imgui.h>
#include <mutex>
#include <thread>

Config       g_config;
Scene *      g_scene     = Scene::create();
Gizmo_Layer *gizmo_layer = NULL;

class Event_Consumer : public IGUI_Pass {
  public:
  void init(rd::Pass_Mng *pmng) override { //
    IGUI_Pass::init(pmng);
  }
  void init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("camera")) {
        gizmo_layer->get_camera().traverse(l->next);
      } else if (l->cmp_symbol("config")) {
        g_config.traverse(l->next);
      } else if (l->cmp_symbol("scene")) {
        g_scene->restore(l);
      }
    }
  }

  void on_gui_traverse_nodes(List *l, int &id) {
    if (l == NULL) return;
    id++;
    ImGui::PushID(id);
    defer(ImGui::PopID());
    if (l->child) {
      ImGui::Indent();
      on_gui_traverse_nodes(l->child, id);
      ImGui::Unindent();
      on_gui_traverse_nodes(l->next, id);
    } else {
      if (l->next == NULL) return;
      if (l->cmp_symbol("scene")) {
        on_gui_traverse_nodes(l->next, id);
        return;
      } else if (l->cmp_symbol("node")) {
        ImGui::LabelText("Node", "%.*s", STRF(l->get(1)->symbol));
        on_gui_traverse_nodes(l->get(2), id);
        return;
      }

      string_ref  type = l->next->symbol;
      char const *name = stref_to_tmp_cstr(l->symbol);
      if (type == stref_s("float3")) {
        float x    = l->get(2)->parse_float();
        float y    = l->get(3)->parse_float();
        float z    = l->get(4)->parse_float();
        float f[3] = {x, y, z};
        if (ImGui::DragFloat3(name, (float *)&f[0], 1.0e-2f)) {
          // if (f[0] != x) {
          // DebugBreak();
          //}
          l->get(2)->symbol = tmp_format("%f", f[0]);
          l->get(3)->symbol = tmp_format("%f", f[1]);
          l->get(4)->symbol = tmp_format("%f", f[2]);
        }
      } else if (type == stref_s("model")) {
        ImGui::LabelText("model", stref_to_tmp_cstr(l->get(2)->symbol));
      } else {
        UNIMPLEMENTED;
      }
    }
  }

  void on_gui(rd::IFactory *factory) override { //
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
      g_scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        g_scene->restore(cur);
      }
    }
    ImGui::End();
    ImGui::Begin("Config");
    g_config.on_imgui();
    ImGui::End();
  }
  void on_init(rd::IFactory *factory) override { //
    TMP_STORAGE_SCOPE;
    gizmo_layer = Gizmo_Layer::create(factory);
    // new XYZDragGizmo(gizmo_layer, &pos);
    g_config.init(stref_s(R"(
(
 (add u32  g_buffer_width 512 (min 4) (max 1024))
 (add u32  g_buffer_height 512 (min 4) (max 1024))
)
)"));

    // g_scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/monkey.gltf"));
    g_scene->update();
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
  }
  void on_release(rd::IFactory *factory) override { //
    // thread_pool.release();
    FILE *scene_dump = fopen("scene_state", "wb");
    fprintf(scene_dump, "(\n");
    defer(fclose(scene_dump));
    gizmo_layer->get_camera().dump(scene_dump);
    g_config.dump(scene_dump);
    {
      String_Builder sb;
      sb.init();
      g_scene->save(sb);
      fwrite(sb.get_str().ptr, 1, sb.get_str().len, scene_dump);
      sb.release();
    }
    fprintf(scene_dump, ")\n");
    gizmo_layer->release();
    g_scene->release();
    IGUI_Pass::release(factory);
  }
  void consume(void *_event) override { //
    IGUI_Pass::consume(_event);
  }
  void on_frame(rd::IFactory *factory) override { //
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GizmoComponent>() == NULL) {
          GizmoComponent::create(gizmo_layer, mn);
        }
      }
    });
    g_scene->get_root()->update();
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
