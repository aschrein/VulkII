
#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <atomic>
//#include <functional>
#include <3rdparty/half.hpp>
#include <imgui.h>
#include <mutex>
#include <thread>

Config       g_config;
Scene *      g_scene     = Scene::create();
Gizmo_Layer *gizmo_layer = NULL;

struct SDF_float8x8 {
  f32 sdf[64];
}; // 32 bits / voxel

struct SDF_half8x8 {
  half_float::half sdf[64];
}; // 16 bits / voxel

struct SDF_byte8x8 {
  i8 sdf[64];
}; // 8 bits / voxel

struct SDF_diff8x8 {
  f32 min;
  f32 max;       // min + max = 64 bits
  u8  grade[64]; // 64 bits
};               // 9 bits / voxel => 1 bit overhead per voxel

struct SDF_diff4x4 {
  f32 min;
  f32 max;
  u8  grade[16];
}; // 6 bits / voxel

struct SDF_Node {
  bool      is_leaf;
  SDF_Node *children[8];
  float     traverse(float3 const &p, float3 const &min, float3 const &max) {
    if (is_leaf) {
    }
  }
};

struct SDF_float8x8_Volume {
  SDF_float8x8 *blocks = NULL;
  int3          size;
  float3        min, max;
  float         voxel_size;
  float         block_size;

  void release() {
    if (blocks) delete[] blocks;
    MEMZERO(*this);
  }

  ~SDF_float8x8_Volume() { release(); }

  void dump(string_ref path) {
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "wb");
    fwrite(&size, 1, sizeof(size), f);
    fwrite(&min, 1, sizeof(min), f);
    fwrite(&max, 1, sizeof(max), f);
    fwrite(&voxel_size, 1, sizeof(voxel_size), f);
    fwrite(&block_size, 1, sizeof(block_size), f);
    fwrite(blocks, 1, sizeof(SDF_float8x8) * size.x * size.y * size.z, f);
    fclose(f);
  }

  bool restore(string_ref path) {
    release();
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "rb");
    if (f == NULL) return false;
    fread(&size, 1, sizeof(size), f);
    fread(&min, 1, sizeof(min), f);
    fread(&max, 1, sizeof(max), f);
    fread(&voxel_size, 1, sizeof(voxel_size), f);
    blocks = new SDF_float8x8[size.x * size.y * size.z];
    fread(blocks, 1, sizeof(SDF_float8x8) * size.x * size.y * size.z, f);
    fclose(f);
    return true;
  }
};

struct Raw_SDF_Volume {
  f32 *  sdf = NULL;
  int3   size;
  float3 min, max;
  float  voxel_size;

  f32 fetch(int3 p) {
    if (p.x >= size.x || p.x < 0 || p.y >= size.y || p.y < 0) return voxel_size;
    return sdf[p.x + p.y * size.x + p.z * size.x * size.y];
  }

  void dump(string_ref path) {
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "wb");
    fwrite(&size, 1, sizeof(size), f);
    fwrite(&min, 1, sizeof(min), f);
    fwrite(&max, 1, sizeof(max), f);
    fwrite(&voxel_size, 1, sizeof(voxel_size), f);
    fwrite(sdf, 1, sizeof(f32) * size.x * size.y * size.z, f);
    fclose(f);
  }

  bool restore(string_ref path) {
    release();
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "rb");
    if (f == NULL) return false;
    fread(&size, 1, sizeof(size), f);
    fread(&min, 1, sizeof(min), f);
    fread(&max, 1, sizeof(max), f);
    fread(&voxel_size, 1, sizeof(voxel_size), f);
    sdf = new f32[size.x * size.y * size.z];
    fread(sdf, 1, sizeof(f32) * size.x * size.y * size.z, f);
    fclose(f);
    return true;
  }

  SDF_float8x8_Volume *generate_blocks() {
    ASSERT_ALWAYS((size.x & 7) == 0);
    ASSERT_ALWAYS((size.y & 7) == 0);
    ASSERT_ALWAYS((size.z & 7) == 0);
    int3                 new_size = int3(size.x / 8, size.y / 8, size.z / 8);
    SDF_float8x8_Volume *out      = new SDF_float8x8_Volume;
    out->size                     = new_size;
    out->blocks                   = new SDF_float8x8[new_size.x * new_size.y * new_size.z];
    out->block_size               = voxel_size * 8;
    out->min                      = min;
    out->max                      = max;
    out->voxel_size               = voxel_size;
    zto(new_size.z) {
      yto(new_size.y) {
        xto(new_size.x) {
          SDF_float8x8 block;
          i32          offset_x = x * 8;
          i32          offset_y = y * 8;
          i32          offset_z = z * 8;
          ito(8) {
            jto(8) {
              kto(8) {
                i32 ix                           = offset_x + x * 8;
                i32 iy                           = offset_y + y * 8;
                i32 iz                           = offset_z + z * 8;
                f32 val                          = sdf[ix + iy * size.x + iz * size.x * size.y];
                block.sdf[k + j * 8 + i * 8 * 8] = val;
              }
            }
          }
          out->blocks[x + y * new_size.x + z * new_size.x * new_size.y] = block;
        }
      }
    }
    return out;
  }

  void release() {
    if (sdf) delete[] sdf;
    MEMZERO(*this);
  }
  ~Raw_SDF_Volume() { release(); }
};

struct Raw_SDF_Volume_Loader {
  std::mutex           mutex;
  std::thread *        thread        = NULL;
  Raw_SDF_Volume *     volume        = NULL;
  SDF_float8x8_Volume *volume_blocks = NULL;

  bool isReady() {
    std::lock_guard<std::mutex> lk(mutex);
    if (volume == NULL) return false;
    thread->join();
    delete thread;
    thread = NULL;
    return true;
  }
  void load(string_ref path) {
    thread = new std::thread([this, path] {
      TMP_STORAGE_SCOPE;
      FILE *f = fopen(stref_to_tmp_cstr(path), "rb");
      ASSERT_ALWAYS(f);
      // 144 223 253
      // -54.1739 33.5713 -161.636
      // 0.8
      int   w;
      int   h;
      int   d;
      float minx;
      float miny;
      float minz;
      float dr;

      fscanf(f, "%i %i %i", &w, &h, &d);
      fscanf(f, "%f %f %f", &minx, &miny, &minz);
      fscanf(f, "%f", &dr);
      float *sdf  = new float[w * h * d];
      float *iter = sdf;
      ito(w * h * d) fscanf(f, "%f", iter++);
      fclose(f);
      {
        std::lock_guard<std::mutex> lk(mutex);
        volume = new Raw_SDF_Volume;
        if (!volume->restore(stref_s("raw_sdf.bin"))) {
          volume->sdf        = sdf;
          volume->size       = int3(w, h, d);
          volume->min        = float3(minx, miny, minz);
          volume->max        = float3(minx + dr * w, miny + dr * h, minz + dr * d);
          volume->voxel_size = dr;
          fprintf(stdout, "SDF has been parsed\n");
          volume->dump(stref_s("raw_sdf.bin"));
          fprintf(stdout, "SDF has been dumped\n");
          volume_blocks = volume->generate_blocks();
          fprintf(stdout, "SDF bocks have been generated\n");
          volume_blocks->dump(stref_s("blocks_sdf.bin"));
          fprintf(stdout, "SDF Blocks has been dumped\n");
        } else {
          fprintf(stdout, "SDF has been restored from cache\n");
        }
        if (!volume_blocks) {
          volume_blocks = new SDF_float8x8_Volume;
          bool suc      = volume_blocks->restore(stref_s("blocks_sdf.bin"));
          ASSERT_ALWAYS(suc);
        }
      }
    });
  }
} g_sdf_loader;

class GBufferPass {
  public:
  Resource_ID normal_rt;
  Resource_ID depth_rt;

  public:
  void init() { MEMZERO(*this); }
  void render(rd::IFactory *factory) {
    float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f, 0.0f));
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GfxSufraceComponent>() == NULL) {
          GfxSufraceComponent::create(factory, mn);
        }
        render_bvh(bvh_visualizer_offset, mn->getComponent<GfxSufraceComponent>()->getBVH(),
                   gizmo_layer);
      }
    });

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
      rt0.clear_color.r     = 0.5f;
      rt0.clear_color.g     = 0.5f;
      rt0.clear_color.b     = 0.5f;
      rt0.clear_color.a     = 1.0f;
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
  @(EXPORT_POSITION mul4(viewproj, mul4(world_transform, float4(POSITION, 1.0))));
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
      float4x4 viewproj = gizmo_layer->get_camera().viewproj();
      ctx->push_constants(&viewproj, 0, sizeof(float4x4));
      g_scene->traverse([&](Node *node) {
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
  }
  void release(rd::IFactory *factory) { factory->release_resource(normal_rt); }
};

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
    g_sdf_loader.load(stref_s("models/light/source/d5c44c10fd0844fabbb8ddb7a25ad77f.sdf"));
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
