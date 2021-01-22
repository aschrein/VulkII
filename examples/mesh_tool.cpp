#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC
//#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "scene.hpp"
#include "script.hpp"

#include <atomic>
//#include <functional>
#include <3rdparty/half.hpp>
#include <condition_variable>
#include <imgui.h>
#include <mutex>
#include <thread>

#include <embree3/rtcore_builder.h>

struct RenderingContext {
  rd::IDevice *factory     = NULL;
  Config *     config      = NULL;
  Scene *      scene       = NULL;
  Gizmo_Layer *gizmo_layer = NULL;
  u32          frame_id    = 0;
  void         dump() {
    FILE *scene_dump = fopen("scene_state", "wb");
    fprintf(scene_dump, "(\n");
    defer(fclose(scene_dump));
    if (gizmo_layer) gizmo_layer->get_camera().dump(scene_dump);
    config->dump(scene_dump);
    if (scene) {
      String_Builder sb;
      sb.init();
      scene->save(sb);
      fwrite(sb.get_str().ptr, 1, sb.get_str().len, scene_dump);
      sb.release();
    }
    fprintf(scene_dump, ")\n");
  }
};

float3 force(float b, float c, float3 v) {
  float len = dot(v, v);
  return b * v;
}

void heal(float3 &p) { ito(3) if (isnan(p[i]) || isinf(p[i]) || fabsf(p[i]) > 1.0e1f) p[i] = 0.0f; }
class RenderPass {
  public:
  static constexpr char const *NAME = "GBuffer Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(color_rt);                                                                              \
  RESOURCE(depth_rt);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  u32 width  = 0;
  u32 height = 0;

  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.init(dev);

    Resource_ID vs = dev->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 color   : TEXCOORD0;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] uint vertex_id : SV_VertexID;
};

float3 random_color(uint id) {
  float2 uv    = frac(float2(float(id) * 15.718281828459045, float(id) * 95.718281828459045));
  float3 seeds = float3(0.123, 0.456, 0.789);
  seeds        = frac((uv.x + 0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.x, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  seeds        = frac((uv.y + 0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.y, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  seeds        = frac((0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.x, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  return seeds;
}

PSInput main(in VSInput input) {
  PSInput output;
  output.color  = random_color(input.vertex_id);
  output.pos    = mul(pc.viewproj, mul(pc.world_transform, float4(input.pos, 1.0f)));
  return output;
}
)"),
                                        NULL, 0);
    Resource_ID ps = dev->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 color   : TEXCOORD0;
};

float4 main(in PSInput input) : SV_TARGET0 {
  return float4(input.color.xyz, 1.0f);
}
)"),
                                        NULL, 0);
    dev->release_resource(vs);
    dev->release_resource(ps);
    signature = [=] {
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
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
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
      gfx_state.VS_set_shader(vs);
      gfx_state.PS_set_shader(ps);
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
      /* {
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
       }*/
      gfx_state.IA_set_vertex_binding(0, 12, rd::Input_Rate::VERTEX);
      // gfx_state.IA_set_vertex_binding(1, 12, rd::Input_Rate::VERTEX);
      // gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();
  }
  void update_frame_buffer(RenderingContext rctx) {
    auto dev = rctx.factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (color_rt.is_valid()) dev->release_resource(color_rt);
    if (depth_rt.is_valid()) dev->release_resource(depth_rt);

    color_rt = [=] {
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
      rt0.image  = color_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);

      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render(RenderingContext &rctx) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f,
    // 0.0f));
    // bthing.test_buffers(dev);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "Render Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      pc.viewproj = viewproj;

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->push_constants(&viewproj, 0, sizeof(float4x4));
      ctx->bind_table(table);
      ctx->bind_graphics_pso(pso);
      float fric = rctx.config->get_f32("sim.fric", 0.9f, 0.0f, 1.0f);
      float simb = rctx.config->get_f32("sim.b", 0.0f, 0.0f, 1.0f);
      float simc = rctx.config->get_f32("sim.c", 0.0f, 0.0f, 1.0f);
      if (rctx.config->get_bool("sim.enable")) {
        rctx.frame_id++;
        // enable_fpe();
        // defer(disable_fpe());
        if (rctx.frame_id == 1) {
          rctx.scene->traverse([&](Node *node) {
            if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
              if (auto *sc = mn->getComponent<TopomeshComponent>()) {
                ito(sc->get_num_meshes()) {
                  Topo_Mesh *tm = sc->get_topo(i);
                  for (int j = 0; j < (i32)tm->vertices.size; j++) {
                    Topo_Mesh::Vertex &v = tm->vertices[j];
                    v.pos /= mn->getAABB().max_dim() * 0.5f;
                  }
                  // Pin Seam vertices
                  jto(tm->seam_edges.size) {
                    u32                seam_id = tm->seam_edges[j];
                    Topo_Mesh::Edge &  e       = tm->edges[seam_id];
                    Topo_Mesh::Vertex &v0      = tm->vertices[e.origin];
                    Topo_Mesh::Vertex &v1      = tm->vertices[e.end];
                    if (length(v1.pos) > 1.0e-3f) {
                      if (fabsf(v1.pos.x) > fabsf(v1.pos.z)) {
                        // v1.pos = normalize(v1.pos);
                        float ratio = v1.pos.z / v1.pos.x;
                        // v1.pos.x    = (sign(v1.pos.x) * 1.001f) * 0.5f + 0.5f;
                        v1.pos.x = sign(v1.pos.x) * 1.001f;
                        v1.pos.z = v1.pos.x * ratio;
                      } else {
                        float ratio = v1.pos.x / v1.pos.z;
                        v1.pos.z    = sign(v1.pos.z) * 1.001f;
                        v1.pos.x    = v1.pos.z * ratio;
                      }
                      /*   v1.pos.x = (v1.pos.x * 1.001f) * 0.5f + 0.5f;
                         v1.pos.z = (v1.pos.x * 1.001f) * 0.5f + 0.5f;*/
                    }
                    v1.bparam1 = true;
                  }
                  // Translate to 0..1 x 0..1 the usual uv space
                  for (int j = 0; j < (i32)tm->vertices.size; j++) {
                    Topo_Mesh::Vertex &v = tm->vertices[j];
                    v.pos.x = v.pos.x * 0.5f + 0.5f;
                    v.pos.z = v.pos.z * 0.5f + 0.5f;
                  }
                }
              }
            }
          });
        }
        rctx.scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            if (auto *sc = mn->getComponent<TopomeshComponent>()) {
              ito(sc->get_num_meshes()) {
                Topo_Mesh *tm = sc->get_topo(i);
#pragma omp parallel
#pragma omp for
                for (int j = 0; j < (i32)tm->vertices.size; j++) {
                  Topo_Mesh::Vertex v = tm->vertices[j];
                  kto(v.edges.size) {
                    Topo_Mesh::Edge &e        = tm->edges[v.edges[k]];
                    u32              other_id = e.origin;
                    if (other_id == j) other_id = e.end;
                    v.f3param0 += force(simb, simc, tm->vertices[other_id].pos - v.pos);
                  }
                  heal(v.f3param0);
                  if (length(v.f3param0) > 1.0f) v.f3param0 = (v.f3param0);
                  if (!v.bparam1) v.pos += v.f3param0;
                  // if (length(v.pos) > 1.1f) v.pos = float3(0.0f, 0.0f, 0.0f);
                  v.f3param0 *= fric;
                  v.pos.y = 0.0f;
                  heal(v.pos);
                  tm->vertices[j] = v;
                }
              }
            }
          }
        });
      }
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (auto *sc = mn->getComponent<TopomeshComponent>()) {
            ito(sc->get_num_meshes()) {
              Topo_Mesh * tm            = sc->get_topo(i);
              Resource_ID vertex_buffer = [&] {
                rd::Buffer_Create_Info buf_info;
                MEMZERO(buf_info);
                buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
                buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
                buf_info.size        = tm->vertices.size * sizeof(float3);
                return dev->create_buffer(buf_info);
              }();
              {
                float3 *pos                   = (float3 *)dev->map_buffer(vertex_buffer);
                jto(tm->vertices.size) pos[j] = tm->vertices[j].pos;
                dev->unmap_buffer(vertex_buffer);
              }
              dev->release_resource(vertex_buffer);
              Resource_ID index_buffer = [&] {
                rd::Buffer_Create_Info buf_info;
                MEMZERO(buf_info);
                buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
                buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
                buf_info.size        = tm->faces.size * sizeof(u32) * 3;
                return dev->create_buffer(buf_info);
              }();
              {
                u32 *indx = (u32 *)dev->map_buffer(index_buffer);
                jto(tm->faces.size) {
                  indx[j * 3 + 0] = tm->faces[j].vtx0;
                  indx[j * 3 + 1] = tm->faces[j].vtx1;
                  indx[j * 3 + 2] = tm->faces[j].vtx2;
                }
                dev->unmap_buffer(index_buffer);
              }
              dev->release_resource(index_buffer);
              ctx->bind_vertex_buffer(0, vertex_buffer, 0);
              ctx->bind_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
              float4x4 world     = mn->get_transform();
              pc.world_transform = world;
              table->push_constants(&pc, 0, sizeof(pc));
              ctx->draw_indexed(tm->faces.size * 3, 1, 0, 0, 0);
            }
          }
        }
      });
      // rctx.scene->traverse([&](Node *node) {
      //  if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
      //    GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
      //    float4x4             world = mn->get_transform();
      //    pc.world_transform         = world;
      //    table->push_constants(&pc, 0, sizeof(pc));
      //    ito(gs->getNumSurfaces()) {
      //      GfxSurface *s = gs->getSurface(i);
      //      s->draw(ctx, gfx_state);
      //    }
      //  }
      //});

      if (rctx.config->get_bool("gizmo.enable")) {
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

        if (rctx.config->get_bool("gizmo.render_bounds")) {
          rctx.scene->traverse([&](Node *node) {
            AABB     aabb = node->getAABB();
            float4x4 t(1.0f);
            rctx.gizmo_layer->render_linebox(transform(t, aabb.min), transform(t, aabb.max),
                                             float3(1.0f, 0.0f, 0.0f));
          });
        }
        if (rctx.config->get_bool("gizmo.render_seams")) {
          rctx.scene->traverse([&](Node *node) {
            if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
              if (auto *sc = mn->getComponent<TopomeshComponent>()) {
                ito(sc->get_num_meshes()) {
                  Topo_Mesh *tm = sc->get_topo(i);
                  jto(tm->seam_edges.size) {
                    u32               seam_id = tm->seam_edges[j];
                    Topo_Mesh::Edge   e       = tm->edges[seam_id];
                    Topo_Mesh::Vertex v0      = tm->vertices[e.origin];
                    Topo_Mesh::Vertex v1      = tm->vertices[e.end];
                    rctx.gizmo_layer->draw_line(v0.pos, v1.pos, float3(1.0f, 1.0f, 0.0f));
                    if (rctx.config->get_bool("gizmo.render_seams.numbers")) {
                      static char buf[0x100];
                      snprintf(buf, sizeof(buf), "%i", j);
                      rctx.gizmo_layer->draw_string(stref_s(buf), v0.pos, float3(1.0f, 1.0f, 1.0f));
                    }
                  }
                }
              }
            }
          });
        }
        if (rctx.config->get_bool("gizmo.render_bvh")) {
          rctx.scene->traverse([&](Node *node) {
            if (MeshNode *mn = node->dyn_cast<MeshNode>()) {

              if (auto *sc = mn->getComponent<GfxSufraceComponent>()) {
                if (sc->getBVH()) {
                  render_bvh(float4x4(1.0f), mn->getComponent<GfxSufraceComponent>()->getBVH(),
                             rctx.gizmo_layer);
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

      float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();

      rctx.gizmo_layer->render(ctx, width, height);
      rctx.gizmo_layer->reset();

      ctx->end_render_pass();
      timestamps.end_range(ctx);
    }

    Resource_ID e = dev->end_render_pass(ctx);
    timestamps.commit(e);
  }
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};

class Event_Consumer : public IGUIApp {
  public:
  RenderPass render_pass;

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
    if (rctx.config->on_imgui()) rctx.dump();
    ImGui::Text("%s %fms", render_pass.get_duration().second, render_pass.get_duration().first);
    if (ImGui::Button("Rebuild BVH")) {
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (mn->getComponent<GfxSufraceComponent>()) {
            mn->getComponent<GfxSufraceComponent>()->buildBVH();
          }
        }
      });
    }
    if (ImGui::Button("Reset topomesh")) {
      rctx.frame_id = 0;
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (mn->getComponent<TopomeshComponent>()) {
            mn->getComponent<TopomeshComponent>()->release();
            TopomeshComponent::create(mn);
          }
        }
      });
    }
    ImGui::LabelText("", "frame id: %i", rctx.frame_id);
    if (ImGui::Button("Save OBJ")) {
      FILE *f = fopen("dump.obj", "wb");
      defer(fclose(f));
      ASSERT_DEBUG(f);
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (auto *sc = mn->getComponent<TopomeshComponent>()) {
            ito(sc->get_num_meshes()) {
              Topo_Mesh *tm   = sc->get_topo(i);
              auto       surf = mn->getSurface(i);
              jto(tm->vertices.size) {
                auto   vtx = surf->mesh.fetch_vertex(j);
                float2 uv  = tm->vertices[j].pos.xz;
                fprintf(f, "v %f %f %f\n", vtx.position.x, vtx.position.y, vtx.position.z);
                fprintf(f, "vn %f %f %f\n", vtx.normal.x, vtx.normal.y, vtx.normal.z);
                fprintf(f, "vt %f %f\n", uv.x, uv.y);
              }
              jto(tm->faces.size) {
                auto face = tm->faces[j];
                fprintf(f, "f %i/%i/%i %i/%i/%i %i/%i/%i\n", face.vtx0 + 1, face.vtx0 + 1,
                        face.vtx0 + 1, face.vtx1 + 1, face.vtx1 + 1, face.vtx1 + 1, face.vtx2 + 1,
                        face.vtx2 + 1, face.vtx2 + 1);
              }
            }
          }
        }
      });
    }
    ImGui::End();

    ImGui::Begin("Render Pass");

    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(render_pass.color_rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
      { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    }
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
  (add u32  g_buffer_width 512 (min 4) (max 2048))
  (add u32  g_buffer_height 512 (min 4) (max 2048))
 )
 )"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/cut.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/norradalur-froyar/scene.gltf"));
    rctx.scene->load_mesh(stref_s("mesh"),
                          stref_s("models/castle-ban-the-rhins-of-galloway/scene.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/untitled.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
    rctx.scene->update();
    rctx.scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GfxSufraceComponent>() == NULL) {
          GfxSufraceComponent::create(rctx.factory, mn);
          TopomeshComponent::create(mn);
        }
      }
    });
    render_pass.init(rctx);

    rctx.gizmo_layer = Gizmo_Layer::create(factory, render_pass.pass);
    char *state      = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
  }
  void on_release() override { //
    rctx.dump();
    rctx.gizmo_layer->release();
    rctx.scene->release();
    rctx.config->release();
    render_pass.release(rctx.factory);

    delete rctx.config;
  }
  void on_frame() override { //
    rctx.scene->get_root()->update();
    render_pass.render(rctx);
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto window_loop = [](rd::Impl_t impl) { IGUIApp::start<Event_Consumer>(impl); };
  // std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  // std::thread dx12_thread = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  // vulkan_thread.join();
  // dx12_thread.join();

  window_loop(rd::Impl_t::DX12);
  return 0;
}