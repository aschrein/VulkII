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

struct Topo_Mesh {
  struct Vertex;
  struct Edge;
  struct TriFace {
    u32 edge_0;
    u32 edge_1;
    u32 edge_2;

    u32 vtx0;
    u32 vtx1;
    u32 vtx2;

    void init() { memset(this, 0, sizeof(TriFace)); }
    void release() {}
  };
  struct Edge {
    u32 origin;
    u32 end;
    u32 face;
    i32 sibling;
    u32 next_edge;
    u32 prev_edge;

    void init() { memset(this, 0, sizeof(Edge)); }
    void release() {}
  };
  struct Vertex {
    SmallArray<u32, 8> edges;
    SmallArray<u32, 8> faces;
    u32                index;
    float3             pos;

    void init() {
      memset(this, 0, sizeof(Vertex));
      edges.init();
      faces.init();
    }
    void release() {
      edges.release();
      faces.release();
    }
  };
  Array<TriFace> faces;
  Array<Edge>    edges;
  // Hash_Table<Pair<u32, u32>, u32, Default_Allocator, 1 << 18, 16> edge_map;
  Array<Vertex> vertices;
  Array<u32>    seam_edges;
  Array<u32>    nonmanifold_edges;
  Edge *        get_edge(u32 id) { return &edges[id]; }
  u32           add_edge() {
    edges.push({});
    return edges.size - 1;
  }
  TriFace *get_face(u32 id) { return &faces[id]; }
  u32      add_face() {
    faces.push({});
    return faces.size - 1;
  }
  Vertex *get_vertex(u32 id) { return &vertices[id]; }
  // void    register_edge(u32 vtx0, u32 vtx1, u32 edge_id) {
  //  //ASSERT_ALWAYS(edge_map.contains({vtx0, vtx1}) == false);
  //  edge_map.insert({vtx0, vtx1}, edge_id);
  //}
  void init(Raw_Mesh_Opaque const &opaque_mesh) {
    // edge_map.init();
    seam_edges.init();
    nonmanifold_edges.init();
    faces.init();
    faces.reserve(opaque_mesh.num_indices / 3);
    edges.init();
    edges.reserve(opaque_mesh.num_indices);
    vertices.init();
    vertices.resize(opaque_mesh.num_vertices);
    vertices.memzero();
    ito(opaque_mesh.num_vertices) {
      vertices[i].index = i;
      vertices[i].pos   = opaque_mesh.fetch_position(i);
    }
    // edge_map.reserve(opaque_mesh.num_indices);
    ito(opaque_mesh.num_indices / 3) {
      Tri_Index face    = opaque_mesh.get_tri_index(i);
      Vertex *  vtx0    = &vertices[face.i0];
      Vertex *  vtx1    = &vertices[face.i1];
      Vertex *  vtx2    = &vertices[face.i2];
      u32       face_id = add_face();
      u32       e0      = add_edge();
      u32       e1      = add_edge();
      u32       e2      = add_edge();

      // register_edge(face.i0, face.i1, e0);
      // register_edge(face.i1, face.i2, e1);
      // register_edge(face.i2, face.i0, e2);

      vtx0->edges.push(e0);
      vtx0->edges.push(e2);
      vtx0->faces.push(face_id);

      vtx1->edges.push(e0);
      vtx1->edges.push(e1);
      vtx1->faces.push(face_id);

      vtx2->edges.push(e1);
      vtx2->edges.push(e2);
      vtx2->faces.push(face_id);

      get_edge(e0)->origin    = face.i0;
      get_edge(e0)->face      = face_id;
      get_edge(e0)->end       = face.i1;
      get_edge(e0)->next_edge = e1;
      get_edge(e0)->prev_edge = e2;
      get_edge(e0)->sibling   = -1;

      get_edge(e1)->origin    = face.i1;
      get_edge(e1)->face      = face_id;
      get_edge(e1)->end       = face.i2;
      get_edge(e1)->next_edge = e2;
      get_edge(e1)->prev_edge = e0;
      get_edge(e1)->sibling   = -1;

      get_edge(e2)->origin    = face.i2;
      get_edge(e2)->face      = face_id;
      get_edge(e2)->end       = face.i0;
      get_edge(e2)->next_edge = e0;
      get_edge(e2)->prev_edge = e1;
      get_edge(e2)->sibling   = -1;

      get_face(face_id)->edge_0 = e0;
      get_face(face_id)->edge_1 = e1;
      get_face(face_id)->edge_2 = e2;
      get_face(face_id)->vtx0   = face.i0;
      get_face(face_id)->vtx1   = face.i1;
      get_face(face_id)->vtx2   = face.i2;
    }
    seam_edges.reserve(edges.size / 10);
    ito(edges.size) {
      Edge *e = get_edge(i);
      // ASSERT_ALWAYS(edge_map.contains({e->origin, e->end}));
      // ASSERT_ALWAYS(edge_map.get({e->origin, e->end}) == i);
      Vertex *dst = get_vertex(e->end);
      jto(dst->edges.size) {
        Edge *se = get_edge(dst->edges[j]);
        if (se->end == e->origin) {
          // ASSERT_DEBUG(e->sibling == -1);
          if (e->sibling != -1) {
            nonmanifold_edges.push(i);
            nonmanifold_edges.push(j);
          }
          e->sibling = dst->edges[j];
        }
      }
      if (e->sibling == -1) {
        seam_edges.push(i);
      }
      // if (edge_map.contains({e->end, e->origin})) {
      //  e->sibling = edge_map.get({e->end, e->origin});
      //} else {
      //  // ASSERT_ALWAYS((*edge_map2).find(std::pair<u32, u32>{e->end,
      //  // e->origin}) == (*edge_map2).end());
      //  seam_edges.push(i);
      //}

      // ASSERT_ALWAYS(edge_map.contains({e->end, e->origin}));
      // ASSERT_ALWAYS(edge_map.get({e->end, e->origin}) != i);
      // e->sibling = edge_map.get({e->end, e->origin});
    }
  }
  void release() {
    ito(edges.size) edges[i].release();
    edges.release();
    ito(faces.size) faces[i].release();
    faces.release();
    ito(vertices.size) vertices[i].release();
    vertices.release();
    // edge_map.release();
    seam_edges.release();
    nonmanifold_edges.release();
  }
};

Hash_Table<u32, Topo_Mesh *> g_topo_meshes;

static int g_init = []() {
  TMP_STORAGE_SCOPE;
  g_camera.init();
  g_config.init(stref_s(R"(
(
 (add bool enable_rasterization_pass 1)
 (add bool draw_wireframe 1)
 (add bool enable_compute_depth 1)
 (add bool enable_compute_render_pass 1)
 (add bool enable_meshlets_render_pass 1)
 (add u32 g_buffer_width 512 (min 4) (max 1024))
 (add u32 g_buffer_height 512 (min 4) (max 1024))
)
)"));

  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
    init_traverse(cur);
  }
  g_scene.init();
  g_scene.load_mesh(stref_s("HIGH"),
                    stref_s("models/human_skull_and_neck/scene_low.gltf"));
  // stref_s("models/low_poly_ellie/scene.gltf"));
  g_topo_meshes.init();
  ito(g_scene.meshes.size) {
    GfxMesh *mesh = g_scene.meshes[i];
    jto(mesh->primitives.size) {
      Primitive &     p        = mesh->primitives[j];
      Raw_Mesh_Opaque new_mesh = optimize_mesh(p.mesh);
      p.mesh.release();
      p.mesh        = new_mesh;
      Topo_Mesh *tm = new Topo_Mesh;
      tm->init(p.mesh);
      ASSERT_DEBUG(p.mesh.id != 0);
      g_topo_meshes.insert(p.mesh.id, tm);
      /*tm.release();*/
    }
  }
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

struct Statistics {
  u32 sampled_texels = 0;
} g_statistics;

class Update_Pass : public rd::IPass {
  public:
  void on_end(rd::IPass_Context *rm) override {}
  void on_begin(rd::IPass_Context *rm) override {
    g_camera.update();
    g_scene.on_pass_begin(rm);
  }
  void       exec(rd::Imm_Ctx *ctx) override {}
  string_ref get_name() override { return stref_s("Update_Pass"); }
  void       release(rd::IPass_Context *rm) override {
    g_scene.release();
    delete this;
  }
};

class Bake_Position_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;

  public:
  Bake_Position_Pass() {
    vs.reset();
    ps.reset();
  }
  void on_end(rd::IPass_Context *rm) override {}
  void on_begin(rd::IPass_Context *rm) override {
    width  = g_config.get_u32("g_buffer_width");
    height = g_config.get_u32("g_buffer_height");
    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.0f;
      cl.g     = 0.0f;
      cl.b     = 0.0f;
      cl.a     = 0.0f;
      rd::Image_Create_Info rt0_info;
      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      rm->add_render_target(stref_s("bake_pass/rt0"), rt0_info, 0, 0, cl);
    }
    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.0f;
      cl.g     = 0.0f;
      cl.b     = 0.0f;
      cl.a     = 0.0f;
      rd::Image_Create_Info rt0_info;
      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      rm->add_render_target(stref_s("bake_pass/rt1"), rt0_info, 0, 0, cl);
    }
    static string_ref            shader    = stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float4x4)  (name world_transform))
)

#ifdef VERTEX

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))
@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_OUTPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_NORMAL  = NORMAL;
  PIXEL_TEXCOORD0  = TEXCOORD0;
  @(EXPORT_POSITION float4(TEXCOORD0 * 2.0 - float2_splat(1.0), 0.0, 1.0));
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_RENDER_TARGET  (location 0))
@(DECLARE_RENDER_TARGET  (location 1))
@(ENTRY)
  @(EXPORT_COLOR 0 float4(PIXEL_POSITION, 1.0));
  @(EXPORT_COLOR 1 float4(PIXEL_NORMAL, 1.0));
@(END)
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null())
      vs = rm->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = rm->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    setup_default_state(ctx, 2);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = false;
    ds_state.enable_depth_write = false;
    ctx->DS_set_state(ds_state);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CCW;
    rs_state.cull_mode    = rd::Cull_Mode::NONE;
    ctx->RS_set_state(rs_state);
    g_scene.gfx_bind(ctx);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    float4x4 viewproj = g_camera.viewproj();
    ctx->push_constants(&viewproj, 0, sizeof(float4x4));
    g_scene.traverse([&](Node *node) {
      if (isa<MeshNode>(node)) {
        GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
        ctx->push_constants(&node->get_transform(), 64, sizeof(float4x4));
        gfxmesh->draw(ctx, g_scene.vertex_buffer,
                      g_scene.mesh_offsets.get(gfxmesh));
      }
    });
  }
  string_ref get_name() override { return stref_s("bake_pass"); }
  void       release(rd::IPass_Context *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
    delete this;
  }
};

class Feedback_Pass : public rd::IPass {
  Resource_ID          vs;
  Resource_ID          ps;
  Resource_ID          ps_wireframe;
  Resource_ID          output_image;
  u32                  width, height;
  Gizmo_Layer          gizmo_layer;
  bool                 gizmo_initialized;
  static constexpr u32 NUM_BUFFERS = 8;
  struct Feedback_Buffer {
    Resource_ID buffer;
    Resource_ID fence;
    bool        in_fly;
    void        init(rd::IPass_Context *rm) {
      MEMZERO(*this);
      in_fly = false;
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = 16;
      buffer        = rm->create_buffer(buf_info);

      fence = rm->get_fence(rd::Fence_Position::PASS_FINISED);
    }
    void release(rd::IPass_Context *rm) {
      if (buffer.is_null() == false) rm->release_resource(buffer);
      MEMZERO(*this);
    }
  };
  InlineArray<Feedback_Buffer, NUM_BUFFERS> feedback_buffers;

  public:
  Feedback_Pass() {
    feedback_buffers.init();
    output_image.reset();
    vs.reset();
    ps.reset();
    ps_wireframe.reset();
    width             = 0;
    height            = 0;
    gizmo_initialized = false;
  }
  void on_end(rd::IPass_Context *rm) override { g_scene.on_pass_end(rm); }
  void on_begin(rd::IPass_Context *pc) override {
    if (gizmo_initialized == false) {
      gizmo_initialized = true;
      gizmo_layer.init(pc);
    }

    rd::Image2D_Info info   = pc->get_swapchain_image_info();
    bool             change = false;
    if (width != g_config.get_u32("g_buffer_width")) {
      change = true;
    }
    if (height != g_config.get_u32("g_buffer_height")) {
      change = true;
    }
    width  = g_config.get_u32("g_buffer_width");
    height = g_config.get_u32("g_buffer_height");
    if (change) {
      if (output_image.is_null() == false) pc->release_resource(output_image);

      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::R32_UINT;
      info.width      = width;
      info.height     = height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                        (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      output_image = pc->create_image(info);
      pc->assign_name(output_image, stref_s("feedback_pass/img0"));
    }

    {
      rd::Clear_Depth cl;
      cl.clear = true;
      cl.d     = 0.0f;
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::D32_FLOAT;
      info.width      = width;
      info.height     = height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      pc->add_depth_target(stref_s("feedback_pass/ds"), info, 0, 0, cl);
    }
    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.2f;
      cl.g     = 0.2f;
      cl.b     = 0.2f;
      cl.a     = 1.0f;
      rd::Image_Create_Info rt0_info;
      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      pc->add_render_target(stref_s("feedback_pass/rt0"), rt0_info, 0, 0, cl);
    }
    static string_ref            shader    = stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float4x4)  (name world_transform))
)

@(DECLARE_BUFFER
  (type WRITE_ONLY)
  (set 0)
  (binding 0)
  (type uint)
  (name out_cnt)
)

@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 1)
  (format R32_UINT)
  (name out_image)
)

#ifdef VERTEX

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))
@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_OUTPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_TEXCOORD0 = TEXCOORD0;
  PIXEL_NORMAL = NORMAL;
  float3 position = POSITION;
  // @(EXPORT_POSITION float4(TEXCOORD0, 0.0, 1.0));
  @(EXPORT_POSITION viewproj * float4(position, 1.0));
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  int2 dim = imageSize(out_image);
  i32 x = i32(0.5 + dim.x * PIXEL_TEXCOORD0.x);
  i32 y = i32(0.5 + dim.y * PIXEL_TEXCOORD0.y + 1.0);
  //image_store(out_image, int2(x, y), float4(1.0, 0.0, 0.0, 1.0));
  imageAtomicAdd(out_image, int2(x, y), 100);
  buffer_atomic_add(out_cnt, 0, 1);

  float4 color = float4_splat(1.0) * (0.5 + 0.5 * dot(PIXEL_NORMAL.rgb, normalize(float3(1.0, 1.0, 1.0))));
  @(EXPORT_COLOR 0 color);
@(END)
#endif
#ifdef PIXEL_WIREFRAME
@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  @(EXPORT_COLOR 0 float4_splat(0.0));
@(END)
#endif

)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
        {stref_s("PIXEL_WIREFRAME"), {}},
    };
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);
    if (ps_wireframe.is_null())
      ps_wireframe =
          pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[2], 1);
    bool found_free = false;
    ito(feedback_buffers.size) {
      if (feedback_buffers[i].in_fly) continue;
      found_free = true;
    }
    if (!found_free && feedback_buffers.isfull() == false) {
      feedback_buffers.push({});
      ASSERT_ALWAYS(feedback_buffers.size != 0);
      feedback_buffers[feedback_buffers.size - 1].init(pc);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    setup_default_state(ctx, 1);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      float dx = 1.0e-1f * g_camera.distance;
      gizmo_layer.draw_sphere(g_camera.look_at, dx * 0.04f,
                              float3{1.0f, 1.0f, 1.0f});
      gizmo_layer.draw_cylinder(g_camera.look_at,
                                g_camera.look_at + float3{dx, 0.0f, 0.0f},
                                dx * 0.04f, float3{1.0f, 0.0f, 0.0f});
      gizmo_layer.draw_cylinder(g_camera.look_at,
                                g_camera.look_at + float3{0.0f, dx, 0.0f},
                                dx * 0.04f, float3{0.0f, 1.0f, 0.0f});
      gizmo_layer.draw_cylinder(g_camera.look_at,
                                g_camera.look_at + float3{0.0f, 0.0f, dx},
                                dx * 0.04f, float3{0.0f, 0.0f, 1.0f});
    }

    g_topo_meshes.iter_pairs([&](u32 mesh_id, Topo_Mesh *tm) {
      ito(tm->seam_edges.size) {
        Topo_Mesh::Edge *  e     = tm->get_edge(tm->seam_edges[i]);
        Topo_Mesh::Vertex *vtx0  = tm->get_vertex(e->origin);
        Topo_Mesh::Vertex *vtx1  = tm->get_vertex(e->end);
        float3             color = float3(1.0f, 0.0f, 0.0f);
        gizmo_layer.draw_line(vtx0->pos, vtx1->pos, color);
      }
      ito(tm->nonmanifold_edges.size) {
        Topo_Mesh::Edge *  e     = tm->get_edge(tm->seam_edges[i]);
        Topo_Mesh::Vertex *vtx0  = tm->get_vertex(e->origin);
        Topo_Mesh::Vertex *vtx1  = tm->get_vertex(e->end);
        float3             color = float3(1.0f, 1.0f, 0.0f);
        gizmo_layer.draw_line(vtx0->pos, vtx1->pos, color);
      }
    });
    g_scene.traverse([&](Node *node) {
      if (isa<MeshNode>(node)) {
        GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
        ito(gfxmesh->primitives.size) {
          Primitive &p      = gfxmesh->primitives[i];
          float3     color  = float3(1.0f, 0.0f, 0.0f);
          float3     cube[] = {
              //
              float3(p.mesh.min.x, p.mesh.min.y, p.mesh.min.z),
              float3(p.mesh.max.x, p.mesh.min.y, p.mesh.min.z),
              float3(p.mesh.max.x, p.mesh.max.y, p.mesh.min.z),
              float3(p.mesh.min.x, p.mesh.max.y, p.mesh.min.z),
              float3(p.mesh.min.x, p.mesh.min.y, p.mesh.max.z),
              float3(p.mesh.max.x, p.mesh.min.y, p.mesh.max.z),
              float3(p.mesh.max.x, p.mesh.max.y, p.mesh.max.z),
              float3(p.mesh.min.x, p.mesh.max.y, p.mesh.max.z),
              //
          };
          gizmo_layer.draw_line(cube[0], cube[1], color);
          gizmo_layer.draw_line(cube[1], cube[2], color);
          gizmo_layer.draw_line(cube[2], cube[3], color);
          gizmo_layer.draw_line(cube[3], cube[0], color);
          gizmo_layer.draw_line(cube[0], cube[4 + 0], color);
          gizmo_layer.draw_line(cube[1], cube[4 + 1], color);
          gizmo_layer.draw_line(cube[2], cube[4 + 2], color);
          gizmo_layer.draw_line(cube[3], cube[4 + 3], color);
          gizmo_layer.draw_line(cube[4 + 0], cube[4 + 1], color);
          gizmo_layer.draw_line(cube[4 + 1], cube[4 + 2], color);
          gizmo_layer.draw_line(cube[4 + 2], cube[4 + 3], color);
          gizmo_layer.draw_line(cube[4 + 3], cube[4 + 0], color);
        }
      }
    });
    /*ito(4) {
      gizmo_layer.draw_cylinder(float3{((i >> 0) & 1) * 2.0f - 1.0f, 0.0f,
                                       ((i >> 1) & 1) * 2.0f - 1.0f},
                                float3{((i >> 0) & 1) * 2.0f - 1.0f, 1.0f,
                                       ((i >> 1) & 1) * 2.0f - 1.0f},
                                1.0e-2f, float3{0.0f, 0.0f, 1.0f});
      gizmo_layer.draw_cone(float3{((i >> 0) & 1) * 2.0f - 1.0f, 1.0f,
                                   ((i >> 1) & 1) * 2.0f - 1.0f},
                            float3{0.0f, 8.0e-2f, 0.0f}, 4.0e-2f,
                            float3{0.0f, 0.0f, 1.0f});
      gizmo_layer.draw_sphere(float3{((i >> 0) & 1) * 2.0f - 1.0f, 0.0f,
                                     ((i >> 1) & 1) * 2.0f - 1.0f},
                              8.0e-2f, float3{0.0f, 0.0f, 1.0f});
    }*/

    {
      rd::Clear_Value cv;
      MEMZERO(cv);
      cv.v_f32[0] = 0.0f;
      cv.v_f32[1] = 0.0f;
      cv.v_f32[2] = 0.0f;
      cv.v_f32[3] = 0.0f;
      ctx->clear_image(output_image, rd::Image_Subresource::top_level(), cv);
      ctx->bind_rw_image(0, 1, 0, output_image,
                         rd::Image_Subresource::top_level(),
                         rd::Format::NATIVE);
    }
    {

      bool found_free = false;
      while (!found_free) {
        ito(feedback_buffers.size) {
          Feedback_Buffer &feedback_buffer = feedback_buffers[i];
          if (feedback_buffer.in_fly &&
              ctx->get_fence_state(feedback_buffer.fence)) {
            u32 *ptr = (u32 *)ctx->map_buffer(feedback_buffer.buffer);
            // fprintf(stdout, "feedback buffer is finished: %i ... \n",
            // ptr[0]);
            g_statistics.sampled_texels = ptr[0];
            ctx->unmap_buffer(feedback_buffer.buffer);
            feedback_buffer.in_fly = false;
          }
        }
        ito(feedback_buffers.size) {
          Feedback_Buffer &feedback_buffer = feedback_buffers[i];
          if (feedback_buffer.in_fly == false) {
            ctx->fill_buffer(feedback_buffer.buffer, 0, 4, 0);
            ctx->bind_storage_buffer(0, 0, feedback_buffer.buffer, 0, 4);
            feedback_buffer.in_fly = true;
            found_free             = true;
            break;
          }
        }
      }
    }
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CCW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    ctx->RS_set_state(rs_state);
    g_scene.gfx_bind(ctx);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    float4x4 viewproj = g_camera.viewproj();
    ctx->push_constants(&viewproj, 0, sizeof(float4x4));
    g_scene.traverse([&](Node *node) {
      if (isa<MeshNode>(node)) {
        GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
        ctx->push_constants(&node->get_transform(), 64, sizeof(float4x4));
        gfxmesh->draw(ctx, g_scene.vertex_buffer,
                      g_scene.mesh_offsets.get(gfxmesh));
      }
    });
    if (g_config.get_bool("draw_wireframe")) {
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::LINE;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      ctx->RS_set_line_width(2.0f);
      ctx->RS_set_depth_bias(1.0e-1f);
      ctx->RS_set_state(rs_state);
      ctx->PS_set_shader(ps_wireframe);
      g_scene.traverse([&](Node *node) {
        if (isa<MeshNode>(node)) {
          GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
          ctx->push_constants(&node->get_transform(), 64, sizeof(float4x4));
          gfxmesh->draw(ctx, g_scene.vertex_buffer,
                        g_scene.mesh_offsets.get(gfxmesh));
        }
      });
    }
    gizmo_layer.render(ctx, g_camera.viewproj());
  }
  string_ref get_name() override { return stref_s("feedback_pass"); }
  void       release(rd::IPass_Context *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
    ito(feedback_buffers.size) feedback_buffers[i].release(rm);
    feedback_buffers.release();
    delete this;
  }
};

class Compute_Rendering_Pass : public rd::IPass {
  Resource_ID output_image;
  Resource_ID position_texture;
  Resource_ID normal_texture;
  Resource_ID output_depth;
  Resource_ID cs;
  u32         width, height;
  Resource_ID sampler;

  public:
  Compute_Rendering_Pass() {
    output_depth.reset();
    cs.reset();
    output_image.reset();
    position_texture.reset();
    normal_texture.reset();
    sampler.reset();
    width  = 0;
    height = 0;
  }
  void on_end(rd::IPass_Context *rm) override {}
  void on_begin(rd::IPass_Context *rm) override {
    position_texture        = rm->get_resource(stref_s("bake_pass/rt0"));
    normal_texture          = rm->get_resource(stref_s("bake_pass/rt1"));
    rd::Image2D_Info info   = rm->get_swapchain_image_info();
    bool             change = false;
    if (width != g_config.get_u32("g_buffer_width")) {
      change = true;
    }
    if (height != g_config.get_u32("g_buffer_height")) {
      change = true;
    }
    width  = g_config.get_u32("g_buffer_width");
    height = g_config.get_u32("g_buffer_height");
    if (change) {
      if (output_image.is_null() == false) rm->release_resource(output_image);
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::RGBA32_FLOAT;
        info.width      = width;
        info.height     = height;
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_image = rm->create_image(info);
        rm->assign_name(output_image, stref_s("Compute_Rendering_Pass/img0"));
      }
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::R32_UINT;
        info.width      = width;
        info.height     = height;
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_depth = rm->create_image(info);
      }
    }
    if (sampler.is_null()) {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
      sampler             = rm->create_sampler(info);
    }
    if (cs.is_null())
      cs = rm->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 1)
  (binding 0)
  (add_field (type float4x4)  (name view))
  (add_field (type float4x4)  (name viewproj))
  (add_field (type uint3)       (name density))
)
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 0)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 1)
  (format R32_UINT)
  (name out_depth)
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 2)
  (format RGBA32_FLOAT)
  (name position_texture)
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 3)
  (format RGBA32_FLOAT)
  (name normal_texture)
)
@(DECLARE_SAMPLER
  (set 0)
  (binding 4)
  (name position_sampler)
)

@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  float2 uv = (float2(GLOBAL_THREAD_INDEX.xy) + float2_splat(0.5)) / float2(density.x, density.y);
  float4 sp = texture(sampler2D(position_texture, position_sampler), uv);
  float3 position = sp.xyz;
  if (sp.a < 1.0)
    return;
  float3 normal = texture(sampler2D(normal_texture, position_sampler), uv).xyz;

  float4 pp = mul4(viewproj, float4(position, 1.0));
  float4 nn = mul4(view, float4(normal, 0.0));
  //nn.xyz / nn.w;

  if (nn.z < 0.0)
    return;

  pp.xyz /= pp.w;
  
  if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
    return;
  i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
  i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
  if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
    float4 color = float4_splat(1.0)
      * (0.5 + 0.5 * dot(normalize(normal), normalize(float3(1.0, 1.0, 1.0))));
    u32 depth = u32(1.0 / pp.z);
    if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
      image_store(out_image, int2(x, y), color);
    }
  }
  
@(END)
)"),
                                 NULL, 0);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    {
      ctx->image_barrier(output_image, (u32)rd::Access_Bits::MEMORY_WRITE,
                         rd::Image_Layout::TRANSFER_DST_OPTIMAL);
      ctx->image_barrier(output_depth, (u32)rd::Access_Bits::MEMORY_WRITE,
                         rd::Image_Layout::TRANSFER_DST_OPTIMAL);
      rd::Clear_Value cv;
      MEMZERO(cv);
      cv.v_f32[0] = 0.0f;
      cv.v_f32[1] = 0.0f;
      cv.v_f32[2] = 0.0f;
      cv.v_f32[3] = 0.0f;
      ctx->clear_image(output_image, rd::Image_Subresource::top_level(), cv);
      MEMZERO(cv);
      cv.v_u32[0] = 1 << 31;
      cv.v_f32[1] = 1 << 31;
      cv.v_f32[2] = 1 << 31;
      cv.v_f32[3] = 1 << 31;
      ctx->clear_image(output_depth, rd::Image_Subresource::top_level(), cv);
      ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                         rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
      ctx->image_barrier(output_depth, (u32)rd::Access_Bits::SHADER_WRITE,
                         rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
      ctx->image_barrier(position_texture, (u32)rd::Access_Bits::SHADER_READ,
                         rd::Image_Layout::SHADER_READ_ONLY_OPTIMAL);
      ctx->image_barrier(normal_texture, (u32)rd::Access_Bits::SHADER_READ,
                         rd::Image_Layout::SHADER_READ_ONLY_OPTIMAL);
    }
    ctx->bind_rw_image(0, 0, 0, output_image,
                       rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    ctx->bind_rw_image(0, 1, 0, output_depth,
                       rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    ctx->bind_image(0, 3, 0, normal_texture, rd::Image_Subresource::top_level(),
                    rd::Format::NATIVE);
    ctx->bind_image(0, 2, 0, position_texture,
                    rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    ctx->bind_sampler(0, 4, sampler);
    ctx->CS_set_shader(cs);
    struct Uniform {
      afloat4x4 view;
      afloat4x4 viewproj;
      auint4    density;
    };

    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
    buf_info.size       = sizeof(Uniform);
    Resource_ID uniform_buffer = ctx->create_buffer(buf_info);
    Uniform *   ptr            = (Uniform *)ctx->map_buffer(uniform_buffer);
    ptr->view                  = g_camera.view;
    ptr->viewproj              = g_camera.viewproj();
    ptr->density.x             = width * 8;
    ptr->density.y             = height * 8;
    ctx->unmap_buffer(uniform_buffer);
    ctx->bind_uniform_buffer(1, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->dispatch((ptr->density.x + 15) / 16, (ptr->density.y + 15) / 16, 1);
    ctx->bind_uniform_buffer(1, 0, Resource_ID::null(), 0, 0);
    ctx->release_resource(uniform_buffer);
  }
  string_ref get_name() override { return stref_s("Compute_Rendering_Pass"); }
  void       release(rd::IPass_Context *rm) override { delete this; }
};

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
      /*if ((m->state & 1) != 0 && last_m_x > 0) {
        i32 dx = cur_m_x - last_m_x;
        i32 dy = cur_m_y - last_m_y;
        g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
        g_camera.theta += (float)(dy)*5.0e-3f;
      }*/
      last_m_x = cur_m_x;
      last_m_y = cur_m_y;
    }
  }
};

class GUI_Pass : public IGUI_Pass {
  Resource_ID staging_buffer;
  Resource_ID copy_fence;
  bool        save_position_buffer;
  u32         save_width;
  u32         save_height;

  public:
  GUI_Pass() {
    save_position_buffer = false;
    staging_buffer.reset();
    copy_fence.reset();
  }
  void on_end(rd::IPass_Context *rm) override { IGUI_Pass::on_end(rm); }
  void on_begin(rd::IPass_Context *rm) override { IGUI_Pass::on_begin(rm); }
  void exec(rd::Imm_Ctx *ctx) override {
    if (save_position_buffer) {
      save_position_buffer = false;
      ctx->copy_image_to_buffer(staging_buffer, 0,
                                ctx->get_resource(stref_s("bake_pass/rt0")),
                                rd::Image_Copy::top_level());
    } else {
      if (copy_fence.is_null() == false) {
        if (ctx->get_fence_state(copy_fence)) {

          void *ptr = ctx->map_buffer(staging_buffer);
          write_image_rgba32_float_pfm("baked_position.pfm", ptr, save_width,
                                       save_height);
          ctx->unmap_buffer(staging_buffer);
          ctx->release_resource(staging_buffer);
          staging_buffer.reset();
          copy_fence.reset();
        }
      }
    }
    IGUI_Pass::exec(ctx);
  }
  void on_gui(rd::IPass_Context *pc) override {
    {
      ImGui::Begin("Config");
      if (ImGui::Button("save position") && staging_buffer.is_null()) {
        rd::Buffer_Create_Info buf_info;
        rd::Image_Info         img_info =
            pc->get_image_info(pc->get_resource(stref_s("bake_pass/rt0")));
        MEMZERO(buf_info);
        buf_info.mem_bits    = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        buf_info.size        = img_info.width * img_info.height * 16;
        staging_buffer       = pc->create_buffer(buf_info);
        copy_fence           = pc->get_fence(rd::Fence_Position::PASS_FINISED);
        save_width           = img_info.width;
        save_height          = img_info.height;
        save_position_buffer = true;
      }
      g_config.on_imgui();
      ImGui::LabelText("sampled_texels", "%i", g_statistics.sampled_texels);
      ImGui::LabelText("hw rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("feedback_pass")));
      ImGui::LabelText(
          "Compute_Rendering_Pass", "%f ms",
          pc->get_pass_duration(stref_s("Compute_Rendering_Pass")));
      ImGui::End();
    }
    {
      ImGui::Begin("main viewport");
      {
        auto wsize = get_window_size();
        ImGui::Image(
            bind_texture(pc->get_resource(stref_s("feedback_pass/rt0")), 0, 0,
                         rd::Format::NATIVE),
            ImVec2(wsize.x, wsize.y));
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
            g_camera.theta -= (float)(dy)*5.0e-3f;
          }
          last_m_x = cur_m_x;
          last_m_y = cur_m_y;
        }
        ImGui::End();
      }
      ImGui::Begin("uv feedback");

      {
        auto wsize = get_window_size();
        ImGui::Image(
            bind_texture(pc->get_resource(stref_s("feedback_pass/img0")), 0, 0,
                         rd::Format::R32_UINT),
            ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();

      ImGui::Begin("position baked");

      {
        auto wsize = get_window_size();
        ImGui::Image(bind_texture(pc->get_resource(stref_s("bake_pass/rt0")), 0,
                                  0, rd::Format::RGBA32_FLOAT),
                     ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
      ImGui::Begin("normals baked");

      {
        auto wsize = get_window_size();
        ImGui::Image(bind_texture(pc->get_resource(stref_s("bake_pass/rt1")), 0,
                                  0, rd::Format::RGBA32_FLOAT),
                     ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
      ImGui::Begin("compute rendering");

      {
        auto wsize = get_window_size();
        ImGui::Image(bind_texture(pc->get_resource(
                                      stref_s("Compute_Rendering_Pass/img0")),
                                  0, 0, rd::Format::RGBA32_FLOAT),
                     ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
    }
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  GUI_Pass *    gui  = new GUI_Pass;
  pmng->set_event_consumer(gui);
  pmng->add_pass(rd::Pass_t::RENDER, new Update_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, new Bake_Position_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, new Feedback_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Compute_Rendering_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, gui);
  pmng->loop();
  return 0;
}
