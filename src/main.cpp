#include "rendering.hpp"
#include "script.hpp"

#include "scene.hpp"

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

#include <imgui.h>
#include <imgui/examples/imgui_impl_sdl.h>

struct Config {
  bool enable_rasterization_pass  = true;
  bool enable_compute_render_pass = true;

  void traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      traverse(l->child);
      traverse(l->next);
    } else {
      if (l->cmp_symbol("enable_rasterization_pass")) {
        enable_rasterization_pass = l->get(1)->parse_int() > 0;
      } else if (l->cmp_symbol("enable_compute_render_pass")) {
        enable_compute_render_pass = l->get(1)->parse_int() > 0;
      }
    }
  }

  void dump(FILE *file) {
    fprintf(file, "(config\n");
    fprintf(file, " (enable_rasterization_pass %i)\n",
            enable_rasterization_pass ? 1 : 0);
    fprintf(file, " (enable_compute_render_pass %i)\n",
            enable_compute_render_pass ? 1 : 0);
    fprintf(file, ")\n");
  }

} g_config;

struct Camera {
  float  phi;
  float  theta;
  float  distance;
  float3 look_at;
  float  aspect;
  float  fov;
  float  znear;
  float  zfar;

  float3   pos;
  float4x4 view;
  float4x4 proj;
  float3   look;
  float3   right;
  float3   up;

  void init() {
    phi      = PI / 2.0f;
    theta    = PI / 2.0f;
    distance = 6.0f;
    look_at  = float3(0.0f, -4.0f, 0.0f);
    aspect   = 1.0;
    fov      = PI / 2.0;
    znear    = 1.0e-3f;
    zfar     = 10.0e5f;
  }

  void traverse(List *l) {
    if (l == NULL)
      return;
    if (l->child) {
      traverse(l->child);
      traverse(l->next);
    } else {
      if (l->cmp_symbol("set_phi")) {
        phi = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_theta")) {
        theta = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_distance")) {
        distance = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_look_at")) {
        look_at.x = l->get(1)->parse_float();
        look_at.y = l->get(2)->parse_float();
        look_at.z = l->get(3)->parse_float();
      } else if (l->cmp_symbol("set_aspect")) {
        aspect = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_fov")) {
        fov = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_znear")) {
        znear = l->get(1)->parse_float();
      } else if (l->cmp_symbol("set_zfar")) {
        zfar = l->get(1)->parse_float();
      }
    }
  }

  void dump(FILE *file) {
    fprintf(file, "(camera\n");
    fprintf(file, " (set_phi %f)\n", phi);
    fprintf(file, " (set_theta %f)\n", theta);
    fprintf(file, " (set_distance %f)\n", distance);
    fprintf(file, " (set_look_at %f %f %f)\n", look_at.x, look_at.y, look_at.z);
    fprintf(file, " (set_aspect %f)\n", aspect);
    fprintf(file, " (set_fov %f)\n", fov);
    fprintf(file, " (set_znear %f)\n", znear);
    fprintf(file, " (set_zfar %f)\n", zfar);
    fprintf(file, ")\n");
  }

  void release() {}

  void update(float2 jitter = float2(0.0f, 0.0f)) {
    pos = float3(sinf(theta) * cosf(phi), cos(theta), sinf(theta) * sinf(phi)) *
              distance +
          look_at;
    look              = normalize(look_at - pos);
    right             = normalize(cross(look, float3(0.0f, 1.0f, 0.0f)));
    up                = normalize(cross(right, look));
    proj              = float4x4(0.0f);
    float tanHalfFovy = std::tan(fov * 0.5f);

    proj[0][0] = 1.0f / (aspect * tanHalfFovy);
    proj[1][1] = 1.0f / (tanHalfFovy);
    proj[2][2] = 0.0f;
    proj[2][3] = -1.0f;
    proj[3][2] = znear;

    proj[2][0] += jitter.x;
    proj[2][1] += jitter.x;
    view = glm::lookAt(pos, look_at, float3(0.0f, 1.0f, 0.0f));
  }
  float4x4 viewproj() { return proj * view; }
} g_camera;

static void init_traverse(List *l) {
  if (l == NULL)
      return;
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
  struct List_Allocator {
    List *alloc() {
      List *out = (List *)tl_alloc_tmp(sizeof(List));
      return out;
    }
  } list_allocator;
  TMP_STORAGE_SCOPE;
  g_camera.init();
  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    List *cur = List::parse(stref_s(state), list_allocator);

    init_traverse(cur);
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

class GfxMeshNode : public MeshNode {
  protected:
  InlineArray<size_t, 16>    attribute_offsets;
  InlineArray<size_t, 16>    attribute_sizes;
  InlineArray<Attribute, 16> attributes;
  size_t                     total_memory_needed;
  u32                        total_indices;
  size_t                     index_offset;
  void                       init(string_ref name) {
    MeshNode::init(name);
    total_memory_needed = 0;
    total_indices       = 0;
    index_offset        = 0;
    attribute_offsets.init();
    attribute_sizes.init();
    attributes.init();
  }

  public:
  static GfxMeshNode *create(string_ref name) {
    GfxMeshNode *out = new GfxMeshNode;
    out->init(name);
    return out;
  }
  static u64 ID() {
    static char p;
    return (u64)(intptr_t)&p;
  }
  u64  get_type_id() const override { return ID(); }
  void release() override { MeshNode::release(); }
  u32  get_num_indices() {
    if (total_indices == 0) {
      ito(primitives.size) total_indices += primitives[i].mesh.num_indices;
    }
    return total_indices;
  }
  size_t get_needed_memory() {
    if (total_memory_needed == 0) {
      if (primitives.size == 0) return 0;
      ito(primitives[0].mesh.attributes.size) {
        attributes.push(primitives[0].mesh.attributes[i]);
        attribute_offsets.push(0);
        attribute_sizes.push(0);
      }
      ito(primitives.size) {
        jto(primitives[i].mesh.attributes.size) {
          attribute_sizes[j] += primitives[i].mesh.get_attribute_size(j);
        }
      }
      ito(attributes.size) {
        jto(i) { attribute_offsets[i] += attribute_sizes[j]; }
        attribute_offsets[i] =
            rd::IResource_Manager::align_up(attribute_offsets[i]);
        total_memory_needed = attribute_offsets[i] + attribute_sizes[i];
      }
      total_memory_needed =
          rd::IResource_Manager::align_up(total_memory_needed);
      index_offset = total_memory_needed;
      ito(primitives.size) {
        total_memory_needed += primitives[i].mesh.get_bytes_per_index() *
                               primitives[i].mesh.num_indices;
      }
    }
    return total_memory_needed;
  }
  void put_data(void *ptr) {
    InlineArray<size_t, 16> attribute_cursors;
    MEMZERO(attribute_cursors);
    size_t indices_offset = 0;
    ito(primitives.size) {
      jto(primitives[i].mesh.attributes.size) {
        Attribute attribute      = primitives[i].mesh.attributes[j];
        size_t    attribute_size = primitives[i].mesh.get_attribute_size(j);
        memcpy((u8 *)ptr + attribute_offsets[j] + attribute_cursors[j],
               &primitives[i].mesh.attribute_data[0] + attribute.offset,
               attribute_size);
        attribute_cursors[j] += attribute_size;
      }
      size_t index_size = primitives[i].mesh.get_bytes_per_index() *
                          primitives[i].mesh.num_indices;
      memcpy((u8 *)ptr + index_offset + indices_offset,
             &primitives[i].mesh.index_data[0], index_size);
      indices_offset += index_size;
    }
  }
  void draw(rd::Imm_Ctx *ctx, Resource_ID vertex_buffer, size_t offset) {
    ito(attributes.size) {
      Attribute attr = attributes[i];
      ctx->IA_set_vertex_buffer(i, vertex_buffer, offset + attribute_offsets[i],
                                attr.stride, rd::Input_Rate::VERTEX);
      static u32 attribute_to_location[] = {
          0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
      };
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = i;
      info.format   = attr.format;
      info.location = attribute_to_location[(u32)attr.type];
      info.offset   = 0;
      info.type     = attr.type;
      ctx->IA_set_attribute(info);
    }
    ctx->IA_set_index_buffer(vertex_buffer, offset + index_offset,
                             rd::Index_t::UINT32);
    // ctx->bind_storage_buffer(0, 1, instance_buffer, 0,
    //                         cpu_instance_info.size *
    //                         sizeof(Instance_Info));
    u32 vertex_cursor = 0;
    u32 index_cursor  = 0;
    ito(primitives.size) {
      ctx->draw_indexed(primitives[i].mesh.num_indices, 1, index_cursor, 0,
                        vertex_cursor);
      index_cursor += primitives[i].mesh.num_indices;
      vertex_cursor += primitives[i].mesh.num_vertices;
    }
  }
};

class Scene {
  Node *             root;
  Array<Image2D_Raw> images;

  // GFX state cache
  Array<Resource_ID>   gfx_images;
  Array<GfxMeshNode *> meshes;
  Array<size_t>        mesh_offsets;
  bool                 gfx_buffers_initialized;
  u32                  texture_streaming_id;
  Resource_ID          vertex_buffer;
  Resource_ID          staging_buffer;
  Resource_ID          dummy_texture;
  bool                 dummy_initialized;

  friend class SceneFactory;
  class SceneFactory : public IFactory {
    Scene *scene;

public:
    SceneFactory(Scene *scene) : scene(scene) {}
    Node *    add_node(string_ref name) override { return Node::create(name); }
    MeshNode *add_mesh(string_ref name) override {
      GfxMeshNode *model = GfxMeshNode::create(name);
      return model;
    }
    u32 add_image(Image2D_Raw img) override {
      scene->images.push(img);
      return scene->images.size - 1;
    }
  };
  template <typename F> void traverse(F fn, Node *node) {
    fn(node);
    ito(node->get_children().size) { traverse(fn, node->get_children()[i]); }
  }

  public:
  void init() {
    dummy_initialized       = false;
    gfx_buffers_initialized = false;
    texture_streaming_id    = 0;
    meshes.init();
    vertex_buffer.reset();
    dummy_texture.reset();
    staging_buffer.reset();
    gfx_images.init();
    images.init();
    SceneFactory sf(this);
    root = load_gltf_pbr(&sf, stref_s("models/mermaid/scene.gltf"));
  }
  void on_pass_begin(rd::IResource_Manager *rm) {
    if (dummy_texture.is_null()) {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA8_UNORM;
      info.width      = 16;
      info.height     = 16;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                        (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
      dummy_texture = rm->create_image(info);
    }
    if (meshes.size == 0) {
      size_t total_memory = 0;
      traverse([&](Node *node) {
        if (isa<GfxMeshNode>(node)) {
          meshes.push((GfxMeshNode *)node);
        }
      });
      ito(meshes.size) {
        size_t mesh_mem = meshes[i]->get_needed_memory();
        mesh_mem        = rd::IResource_Manager::align_up(mesh_mem);
        mesh_offsets.push(total_memory);
        total_memory += mesh_mem;
      }
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER |
                            (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size = total_memory;
      vertex_buffer = rm->create_buffer(buf_info);
    }
    if (gfx_images.size == 0) {
      if (images.size) {
        gfx_images.resize(images.size);
        ito(images.size) {
          rd::Image_Create_Info info;
          MEMZERO(info);
          info.format     = images[i].format;
          info.width      = images[i].width;
          info.height     = images[i].height;
          info.depth      = 1;
          info.layers     = 1;
          info.levels     = 1;
          info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
          info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
          gfx_images[i] = rm->create_image(info);
        }
      }
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      buf_info.size       = 1 << 14;
      staging_buffer      = rm->create_buffer(buf_info);
    }
  }
  void gfx_exec(rd::Imm_Ctx *ctx) {
    if (dummy_initialized == false) {
      {
        u32 *ptr = (u32 *)ctx->map_buffer(staging_buffer);
        ito(16 * 16) { ptr[i] = 0xff0000ffu; }
        ctx->unmap_buffer(staging_buffer);
        ctx->copy_buffer_to_image(staging_buffer, 0, dummy_texture, 0, 0);
      }
      {
        u8 *ptr = (u8 *)ctx->map_buffer(vertex_buffer);
        ito(meshes.size) { meshes[i]->put_data(ptr + mesh_offsets[i]); }
        ctx->unmap_buffer(vertex_buffer);
      }
    } else if (texture_streaming_id < gfx_images.size) {
      void *ptr = ctx->map_buffer(staging_buffer);
      memcpy(ptr, images[texture_streaming_id].data,
             images[texture_streaming_id].get_size_in_bytes());
      ctx->unmap_buffer(staging_buffer);
      ctx->copy_buffer_to_image(staging_buffer, 0,
                                gfx_images[texture_streaming_id], 0, 0);
      texture_streaming_id++;
    }
    ito(gfx_images.size) {
      if (i < texture_streaming_id) {
        ctx->bind_image(1, 1, i, gfx_images[i], 0, 1, 0, 1);
      } else {
        ctx->bind_image(1, 1, i, dummy_texture, 0, 1, 0, 1);
      }
    }
    ito(meshes.size) { meshes[i]->draw(ctx, vertex_buffer, mesh_offsets[i]); }
  }
  void on_pass_end(rd::IResource_Manager *rm) {
    if (staging_buffer.is_null() == false) {
      rm->release_resource(staging_buffer);
      staging_buffer.reset();
    }
  }
  void release_gfx(rd::IResource_Manager *rm) {
    ito(gfx_images.size) rm->release_resource(gfx_images[i]);
    gfx_images.release();
    rm->release_resource(vertex_buffer);
    rm->release_resource(staging_buffer);
  }
  template <typename F> void traverse(F fn) { traverse(fn, root); }
  void                       update() {}
  void                       release() {
    meshes.release();
    ito(images.size) images[i].release();
    images.release();
    root->release();
  }
} g_scene;

static void setup_default_state(rd::Imm_Ctx *ctx, u32 num_rts = 1) {
  rd::Blend_State bs;
  MEMZERO(bs);
  bs.enabled          = false;
  bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT |
                        (u32)rd::Color_Component_Bit::G_BIT |
                        (u32)rd::Color_Component_Bit::B_BIT |
                        (u32)rd::Color_Component_Bit::A_BIT;
  ito(num_rts) ctx->OM_set_blend_state(i, bs);
  ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
  rd::RS_State rs_state;
  MEMZERO(rs_state);
  rs_state.polygon_mode = rd::Polygon_Mode::FILL;
  rs_state.front_face   = rd::Front_Face::CW;
  rs_state.cull_mode    = rd::Cull_Mode::NONE;
  rs_state.line_width   = 1.0f;
  rs_state.depth_bias   = 0.0f;
  ctx->RS_set_state(rs_state);
  rd::DS_State ds_state;
  MEMZERO(ds_state);
  ds_state.cmp_op             = rd::Cmp::EQ;
  ds_state.enable_depth_test  = false;
  ds_state.enable_depth_write = false;
  ctx->DS_set_state(ds_state);
  rd::MS_State ms_state;
  MEMZERO(ms_state);
  ms_state.sample_mask = 0xffffffffu;
  ms_state.num_samples = 1;
  ctx->MS_set_state(ms_state);
}

class Opaque_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID uniform_buffer;
  Resource_ID texture_sampler;
  struct Uniform {
    afloat4x4 viewproj;
  };
  static_assert(sizeof(Uniform) == 64, "Uniform packing is wrong");

  public:
  Opaque_Pass() {
    vs.reset();
    ps.reset();
    width  = 512;
    height = 512;
    uniform_buffer.reset();
    texture_sampler.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
    g_scene.on_pass_end(rm);
  }
  void set_size(u32 width, u32 height) {
    this->width  = width;
    this->height = height;
  }
  void on_begin(rd::IResource_Manager *pc) override {
    // rd::Image2D_Info info = pc->get_swapchain_image_info();
    //    width                 = info.width;
    // height                = info.height;
    g_camera.aspect = (float)width / height;
    g_camera.update();

    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.0f;
      cl.g     = 0.0f;
      cl.b     = 0.0f;
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
      pc->add_render_target(stref_s("opaque_pass/rt0"), rt0_info, 0, 0, cl);
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
      pc->add_depth_target(stref_s("opaque_pass/ds"), info, 0, 0, cl);
    }
    static string_ref            shader    = stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4) (name viewproj))
)
struct Instance_Info {
  i32 albedo_id;
  i32 arm_id;
  i32 normal_id;
  i32 pad;
  float4x4 model;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type Instance_Info)
  (name instance_infos)
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (array_size 1024)
  (dim 2D)
  (set 1)
  (binding 1)
  (name material_textures)
)
@(DECLARE_SAMPLER
  (set 1)
  (binding 0)
  (name my_sampler)
)
#ifdef VERTEX
@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))
@(DECLARE_INPUT (location 2) (type float3) (name BINORMAL))
@(DECLARE_INPUT (location 3) (type float3) (name TANGENT))
@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))
@(DECLARE_INPUT (location 5) (type float2) (name TEXCOORD1))
@(DECLARE_INPUT (location 6) (type float2) (name TEXCOORD2))
@(DECLARE_INPUT (location 7) (type float2) (name TEXCOORD3))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_OUTPUT (location 2) (type float3) (name PIXEL_BINORMAL))
@(DECLARE_OUTPUT (location 3) (type float3) (name PIXEL_TANGENT))
@(DECLARE_OUTPUT (location 4) (type float2) (name PIXEL_TEXCOORD0))
@(DECLARE_OUTPUT (location 5) (type uint) (name PIXEL_INSTANCE_ID))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_NORMAL    = NORMAL;
  PIXEL_BINORMAL  = BINORMAL;
  PIXEL_TANGENT   = TANGENT;
  PIXEL_TEXCOORD0 = TEXCOORD0;
  PIXEL_INSTANCE_ID = INSTANCE_INDEX;
  float3 position = POSITION;
  position.xyz *= 0.04;
  // float4x4 world_matrix = buffer_load(instance_infos, INSTANCE_INDEX).model;
  @(EXPORT_POSITION
      viewproj * float4(position, 1.0)
  );
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float3) (name PIXEL_BINORMAL))
@(DECLARE_INPUT (location 3) (type float3) (name PIXEL_TANGENT))
@(DECLARE_INPUT (location 4) (type float3) (name PIXEL_TEXCOORD0))
@(DECLARE_INPUT (location 5) (type "flat uint") (name PIXEL_INSTANCE_ID))

@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  float4 albedo = float4(0.0, 1.0, 1.0, 1.0);
  /*i32 albedo_id = buffer_load(instance_infos, instance_index).albedo_id;
  if (albedo_id >= 0) {
    albedo = texture(sampler2D(material_textures[nonuniformEXT(albedo_id)], my_sampler), tex_coords);
  }*/
  @(EXPORT_COLOR 0 float4(0.5 * PIXEL_NORMAL.rgb + float3_splat(0.5), 1.0));
@(END)
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = sizeof(Uniform);
      uniform_buffer      = pc->create_buffer(buf_info);
    }
    if (texture_sampler.is_null()) {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
      info.max_anisotropy = 16.0f;
      texture_sampler     = pc->create_sampler(info);
    }
    g_scene.on_pass_begin(pc);
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.enable_rasterization_pass == false) return;
    setup_default_state(ctx);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      Uniform *ptr  = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->viewproj = g_camera.viewproj();
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_sampler(1, 0, texture_sampler);
    g_scene.gfx_exec(ctx);
  }
  string_ref get_name() override { return stref_s("opaque_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
  }
};

class Compute_Render_Pass : public rd::IPass {
  Resource_ID cs;
  u32         width, height;
  Resource_ID output_image;

  public:
  Compute_Render_Pass() {
    cs.reset();
    output_image.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {}
  void on_begin(rd::IResource_Manager *pc) override {
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4)   (name color))
  (add_field (type u32)      (name control_flags))
)
#define CONTROL_ENABLE_FEEDBACK 1
bool is_control_set(uint bits) {
  return (control_flags & bits) != 0;
}
@(DECLARE_IMAGE
  (type READ_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name my_image)
)
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 2)
  (format RGBA32_FLOAT)
  (name out_image)
)
struct BVH_Node {
  float3 min;
  float3 max;
  u32    flags;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 3)
  (type BVH_Node)
  (name bvh_nodes)
)
struct Dummy {
  uint4    flags;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 4)
  (type Dummy)
  (name dummy)
)
float4 op_laplace(int2 coords) {
  float4 val00 = image_load(my_image, coords + int2(-1, 0));
  float4 val01 = image_load(my_image, coords + int2(1, 0));
  float4 val10 = image_load(my_image, coords + int2(0, -1));
  float4 val11 = image_load(my_image, coords + int2(0, 1));
  float4 center = image_load(my_image, coords);
  float4 laplace = abs(center * 4.0 - val00 - val01 - val10 - val11) / 4.0;
  return laplace;
  // float intensity = dot(laplace, float4_splat(1.0)); 
  // return intensity > 0.5 ? float4_splat(1.0) : float4_splat(0.0);
}

@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(my_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  float2 uv = GLOBAL_THREAD_INDEX.xy / dim.xy;
  
  float4 in_val = image_load(my_image, GLOBAL_THREAD_INDEX.xy);
  in_val = pow(in_val, float4(1.0/2.2));
  if (GLOBAL_THREAD_INDEX.y == 777 && is_control_set(CONTROL_ENABLE_FEEDBACK)) {
    Dummy d;
    d.flags.x = GLOBAL_THREAD_INDEX.x;
    d.flags.y = GLOBAL_THREAD_INDEX.y;
    d.flags.z = GLOBAL_THREAD_INDEX.z;
    d.flags.w = 666;
    buffer_store(dummy, GLOBAL_THREAD_INDEX.x, d);
  }
  image_store(out_image, GLOBAL_THREAD_INDEX.xy, in_val);
@(END)
)"),
                                 NULL, 0);
    rd::Image2D_Info info = pc->get_swapchain_image_info();
    if (output_image.is_null() || width != info.width ||
        height != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      width  = info.width;
      height = info.height;
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
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      output_image = pc->create_image(info);
      pc->assign_name(output_image, stref_s("compute_render/img0"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    ctx->CS_set_shader(cs);
    ctx->bind_rw_image(0, 0, 0, output_image, 0, 1, 0, 1);
    ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
  }
  string_ref get_name() override { return stref_s("compute_render_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(output_image);
  }
};

#if 0
				


struct Indirect_Draw_Arguments {
  u32 index_count;
  u32 instance_count;
  u32 first_index;
  i32 vertex_offset;
  u32 first_instance;
};
static_assert(sizeof(Indirect_Draw_Arguments) == 20,
              "Uniform packing is wrong");

class Mesh_Prepare_Pass : public rd::IPass {
  Resource_ID staging_buffer;
  bool        dummy_initialized;
  Resource_ID dummy_texture;
  u32         current_streaming_id;

  Resource_ID vertex_buffer;
  Resource_ID index_buffer;
  Resource_ID instance_buffer;

  Array<Resource_ID> gpu_textures;

  InlineArray<size_t, 16>    attribute_offsets;
  InlineArray<size_t, 16>    attribute_sizes;
  InlineArray<Attribute, 16> attributes;

  struct Instance_Info {
    i32      albedo_id;
    i32      arm_id;
    i32      normal_id;
    i32      dummy;
    float4x4 model;
  };

  static_assert(sizeof(Instance_Info) == 80, "Uniform packing is wrong");

  Array<Instance_Info> cpu_instance_info;

  struct Uniform {
    afloat4x4 viewproj;
  };

  static_assert(sizeof(Uniform) == 64, "Uniform packing is wrong");

  public:
  Array<Indirect_Draw_Arguments> cpu_draw_args;
  Mesh_Prepare_Pass() {
    attribute_offsets.init();
    attribute_sizes.init();
    attributes.init();
    staging_buffer.reset();
    dummy_initialized = false;
    dummy_texture.reset();
    current_streaming_id = 0;
    gpu_textures.init();
    cpu_instance_info.init();
    cpu_draw_args.init();
    vertex_buffer.reset();
    index_buffer.reset();
    instance_buffer.reset();
  }
  void release(rd::IResource_Manager *rm) override {
    attributes.release();
    attribute_sizes.release();
    attribute_offsets.release();
    rm->release_resource(dummy_texture);
    rm->release_resource(staging_buffer);
    ito(gpu_textures.size) rm->release_resource(gpu_textures[i]);
    gpu_textures.release();
    cpu_draw_args.release();
    cpu_instance_info.release();
    rm->release_resource(vertex_buffer);
    rm->release_resource(index_buffer);
    rm->release_resource(instance_buffer);
    delete this;
  }
  void draw(rd::Imm_Ctx *ctx, Resource_ID arg_buf, Resource_ID cnt_buf) {
    ito(gpu_textures.size) {
      if (i < current_streaming_id) {
        ctx->bind_image(1, 1, i, gpu_textures[i], 0, 1, 0, 1);
      } else {
        ctx->bind_image(1, 1, i, dummy_texture, 0, 1, 0, 1);
      }
    }
    static u32 attribute_to_location[] = {
        0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    ito(attributes.size) {
      Attribute attr = attributes[i];
      ctx->IA_set_vertex_buffer(i, vertex_buffer, attribute_offsets[i],
                                attr.stride, rd::Input_Rate::VERTEX);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = i;
      info.format   = attr.format;
      info.location = attribute_to_location[(u32)attr.type];
      info.offset   = 0;
      info.type     = attr.type;
      ctx->IA_set_attribute(info);
    }
    ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
    ctx->bind_storage_buffer(0, 1, instance_buffer, 0,
                             cpu_instance_info.size * sizeof(Instance_Info));
#if 0
    ito(cpu_draw_args.size) {
      Indirect_Draw_Arguments args = cpu_draw_args[i];
      ctx->draw_indexed(args.index_count, args.instance_count,
      args.first_index,
                        args.first_instance, args.vertex_offset);
    }
#endif
    ctx->multi_draw_indexed_indirect(arg_buf, 0, cnt_buf, 0,
                                     cpu_instance_info.size,
                                     sizeof(Indirect_Draw_Arguments));
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (staging_buffer.is_null() == false) {
      rm->release_resource(staging_buffer);
      staging_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    if (dummy_texture.is_null()) {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA8_UNORM;
      info.width      = 16;
      info.height     = 16;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                        (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
      dummy_texture = pc->create_image(info);
    }
    if (gpu_textures.size == 0) {
      gpu_textures.resize(g_scene.model.images.size);
      ito(g_scene.model.images.size) {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = g_scene.model.images[i].format;
        info.width      = g_scene.model.images[i].width;
        info.height     = g_scene.model.images[i].height;
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                          (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
        gpu_textures[i] = pc->create_image(info);
      }
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      buf_info.size       = 1 << 14;
      staging_buffer      = pc->create_buffer(buf_info);
    }
    if (vertex_buffer.is_null()) {

      u32 total_vertex_buffer_size = 0;
      u32 total_index_buffer_size  = 0;
      ito(g_scene.model.meshes.size) {
        total_vertex_buffer_size += g_scene.model.meshes[i].attribute_data.size;
        // for alignment
        total_vertex_buffer_size +=
            g_scene.model.meshes[i].attributes.size * 0x100;
        total_index_buffer_size += g_scene.model.meshes[i].index_data.size;
      }
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
        buf_info.size       = total_index_buffer_size;
        index_buffer        = pc->create_buffer(buf_info);
      }
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
        buf_info.size       = total_vertex_buffer_size;
        vertex_buffer       = pc->create_buffer(buf_info);
      }
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
        buf_info.size       = g_scene.model.meshes.size * sizeof(Instance_Info);
        instance_buffer     = pc->create_buffer(buf_info);
      }
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (dummy_initialized == false) {
      cpu_draw_args.resize(g_scene.model.meshes.size);
      cpu_instance_info.resize(g_scene.model.meshes.size);
      {
        u32 *ptr = (u32 *)ctx->map_buffer(staging_buffer);
        ito(16 * 16) { ptr[i] = 0xff0000ffu; }
        ctx->unmap_buffer(staging_buffer);
        ctx->copy_buffer_to_image(staging_buffer, 0, dummy_texture, 0, 0);
      }
      dummy_initialized = true;
      {
        ito(g_scene.model.meshes[0].attributes.size) {
          attributes.push(g_scene.model.meshes[0].attributes[i]);
          attribute_offsets.push(0);
          attribute_sizes.push(0);
        }
        ito(g_scene.model.meshes.size) {
          {
            Instance_Info instance_info;
            MEMZERO(instance_info);
            instance_info.albedo_id = g_scene.model.materials[i].albedo_id;
            instance_info.arm_id    = g_scene.model.materials[i].arm_id;
            instance_info.normal_id = g_scene.model.materials[i].normal_id;
            instance_info.model =
                rotate(float4x4(1.0f), (float)0.0f, // timer.cur_time,
                       float3(0.0f, 1.0f, 0.0f)) *
                scale(float4x4(1.0f), float3(1.0f, -1.0f, 1.0f) * 0.04f);
            cpu_instance_info[i] = instance_info;
          }
          jto(g_scene.model.meshes[i].attributes.size) {
            ASSERT_ALWAYS(g_scene.model.meshes[0].attributes[j].type ==
                          attributes[j].type);
            ASSERT_ALWAYS(g_scene.model.meshes[0].attributes[j].format ==
                          attributes[j].format);
            attribute_sizes[j] += g_scene.model.meshes[i].get_attribute_size(j);
          }
        }
        ito(attributes.size) {
          jto(i) { attribute_offsets[i] += 0x100 + attribute_sizes[j]; }
        }
        ito(g_scene.model.meshes.size) {
          jto(attributes.size) {
            attribute_offsets[i] =
                rd::IResource_Manager::align_down(attribute_offsets[i]);
          }
        }
        InlineArray<size_t, 16> attribute_cursors;
        MEMZERO(attribute_cursors);
        u8 *ptr = (u8 *)ctx->map_buffer(vertex_buffer);
        ito(g_scene.model.meshes.size) {
          jto(g_scene.model.meshes[i].attributes.size) {
            Attribute attribute = g_scene.model.meshes[i].attributes[j];
            size_t    attribute_size =
                g_scene.model.meshes[i].get_attribute_size(j);
            memcpy(ptr + attribute_offsets[j] + attribute_cursors[j],
                   &g_scene.model.meshes[i].attribute_data[0] +
                       attribute.offset,
                   attribute_size);
            attribute_cursors[j] += attribute_size;
          }
        }
        ctx->unmap_buffer(vertex_buffer);
        Array<u32> vertex_offsets;
        Array<u32> index_offsets;
        vertex_offsets.init(g_scene.model.meshes.size);
        vertex_offsets.resize(g_scene.model.meshes.size);
        vertex_offsets.memzero();
        index_offsets.init(g_scene.model.meshes.size);
        index_offsets.resize(g_scene.model.meshes.size);
        index_offsets.memzero();
        defer({
          index_offsets.release();
          vertex_offsets.release();
        });
        ito(g_scene.model.meshes.size) {
          jto(i) {
            vertex_offsets[i] += g_scene.model.meshes[j].num_vertices;
            index_offsets[i] += g_scene.model.meshes[j].num_indices;
          }
        }
        ito(g_scene.model.meshes.size) {
          Indirect_Draw_Arguments args;
          MEMZERO(args);
          args.first_index    = index_offsets[i];
          args.first_instance = i;
          args.index_count    = g_scene.model.meshes[i].num_indices;
          args.instance_count = 1;
          args.vertex_offset  = vertex_offsets[i];
          cpu_draw_args[i]    = args;
        }
      }
      {
        u8 *   ptr    = (u8 *)ctx->map_buffer(index_buffer);
        size_t offset = 0;
        ito(g_scene.model.meshes.size) {
          memcpy(ptr + offset, &g_scene.model.meshes[i].index_data[0],
                 g_scene.model.meshes[i].index_data.size);
          offset += g_scene.model.meshes[i].index_data.size;
        }
        ctx->unmap_buffer(index_buffer);
      }
      {
        u8 *ptr = (u8 *)ctx->map_buffer(instance_buffer);
        memcpy(ptr, &cpu_instance_info[0],
               cpu_instance_info.size * sizeof(Instance_Info));
        ctx->unmap_buffer(instance_buffer);
      }
    } else if (current_streaming_id < gpu_textures.size) {
      void *ptr = ctx->map_buffer(staging_buffer);
      memcpy(ptr, g_scene.model.images[current_streaming_id].data,
             g_scene.model.images[current_streaming_id].get_size_in_bytes());
      ctx->unmap_buffer(staging_buffer);
      ctx->copy_buffer_to_image(staging_buffer, 0,
                                gpu_textures[current_streaming_id], 0, 0);
      current_streaming_id++;
    }
  }
  string_ref get_name() override { return stref_s("mesh_prepare_pass"); }
};

class Culling_Pass : public rd::IPass {
  Resource_ID cs;
  Resource_ID clear_cs;
  Resource_ID uniform_buffer;

  Timer              timer;
  Mesh_Prepare_Pass *mpass;
  struct Uniform {
    u32 num_args;
  } uniform_data;
  static_assert(sizeof(uniform_data) == 4, "Uniform packing is wrong");

  public:
  Resource_ID in_arg_buf;
  Resource_ID out_arg_buf;
  Resource_ID out_cnt_buf;
  Culling_Pass() {
    cs.reset();
    clear_cs.reset();
    timer.init();
    uniform_buffer.reset();
    in_arg_buf.reset();
    out_arg_buf.reset();
    out_cnt_buf.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    mpass = (Mesh_Prepare_Pass *)pc->get_pass(stref_s("mesh_prepare_pass"));
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type u32)      (name num_meshes))
)
struct Indirect_Draw_Arguments {
  u32 index_count;
  u32 instance_count;
  u32 first_index;
  i32 vertex_offset;
  u32 first_instance;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type Indirect_Draw_Arguments)
  (name in_args)
)
@(DECLARE_BUFFER
  (type WRITE_ONLY)
  (set 0)
  (binding 2)
  (type Indirect_Draw_Arguments)
  (name out_args)
)
@(DECLARE_BUFFER
  (type WRITE_ONLY)
  (set 0)
  (binding 3)
  (type uint)
  (name out_cnt)
)
@(GROUP_SIZE 256 1 1)
@(ENTRY)
  if (GLOBAL_THREAD_INDEX.x > num_meshes)
    return;
  Indirect_Draw_Arguments args = buffer_load(in_args, GLOBAL_THREAD_INDEX.x);
  u32 index = buffer_atomic_add(out_cnt, 0, 1);
  //args.first_instance = index;
  buffer_store(out_args, GLOBAL_THREAD_INDEX.x, args);
@(END)
)"),
                                 NULL, 0);
    if (clear_cs.is_null())
      clear_cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_BUFFER
  (type WRITE_ONLY)
  (set 1)
  (binding 1)
  (type uint)
  (name out_cnt)
)
@(GROUP_SIZE 16 1 1)
@(ENTRY)
  buffer_store(out_cnt, 0, 0);
@(END)
)"),
                                       NULL, 0);
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = sizeof(uniform_data);
      uniform_buffer      = pc->create_buffer(buf_info);
    }
    if (in_arg_buf.is_null() == false) {
      pc->release_resource(in_arg_buf);
      in_arg_buf.reset();
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
      buf_info.size =
          sizeof(Indirect_Draw_Arguments) * g_scene.model.meshes.size;
      in_arg_buf = pc->create_buffer(buf_info);
    }
    if (out_cnt_buf.is_null()) {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      buf_info.usage_bits =
          (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
          (u32)rd::Buffer_Usage_Bits::USAGE_INDIRECT_ARGUMENTS;
      buf_info.size = 16;
      out_cnt_buf   = pc->create_buffer(buf_info);
      pc->assign_name(out_cnt_buf, stref_s("cull/cnt"));
    }
    if (out_arg_buf.is_null()) {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      buf_info.usage_bits =
          (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
          (u32)rd::Buffer_Usage_Bits::USAGE_INDIRECT_ARGUMENTS;
      buf_info.size =
          sizeof(Indirect_Draw_Arguments) * g_scene.model.meshes.size;
      out_arg_buf = pc->create_buffer(buf_info);
      pc->assign_name(out_arg_buf, stref_s("cull/args"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    timer.update();

    {
      Uniform *ptr  = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->num_args = g_scene.model.meshes.size;
      ctx->unmap_buffer(uniform_buffer);
    }
    {
      void *ptr = ctx->map_buffer(in_arg_buf);
      memcpy(ptr, &mpass->cpu_draw_args[0],
             sizeof(Indirect_Draw_Arguments) * mpass->cpu_draw_args.size);
      ctx->unmap_buffer(in_arg_buf);
    }
    ctx->clear_bindings();
    ctx->bind_storage_buffer(1, 1, out_cnt_buf, 0, 4);
    ctx->CS_set_shader(clear_cs);
    ctx->dispatch(1, 1, 1);
    ctx->clear_bindings();

    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_storage_buffer(0, 1, in_arg_buf, 0,
                             sizeof(Indirect_Draw_Arguments) *
                                 g_scene.model.meshes.size);
    ctx->bind_storage_buffer(0, 2, out_arg_buf, 0,
                             sizeof(Indirect_Draw_Arguments) *
                                 g_scene.model.meshes.size);
    ctx->bind_storage_buffer(0, 3, out_cnt_buf, 0, 4);

    ctx->CS_set_shader(cs);
    ctx->dispatch((g_scene.model.meshes.size + 255) / 256, 1, 1);
  }
  string_ref get_name() override { return stref_s("culling_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(clear_cs);
    rm->release_resource(out_arg_buf);
    rm->release_resource(out_cnt_buf);
    if (in_arg_buf.is_null() == false) {
      rm->release_resource(in_arg_buf);
      in_arg_buf.reset();
    }
    timer.release();
  }
};

#endif // 0

class Postprocess_Pass : public rd::IPass {
  Resource_ID cs;
  u32         width, height;
  Resource_ID uniform_buffer;
  Resource_ID sampler;
  Resource_ID input_image;
  Resource_ID output_image;
  struct Feedback_Buffer {
    Resource_ID buffer;
    Resource_ID fence;
    bool        in_fly;
    void        reset() {
      in_fly = false;
      buffer.reset();
      fence.reset();
    }
    void release(rd::IResource_Manager *rm) {
      if (buffer.is_null() == false) rm->release_resource(buffer);
      if (fence.is_null() == false) rm->release_resource(fence);
      reset();
    }
  } feedback_buffer;
  Timer timer;

  struct Uniform {
    afloat4 color;
    ai32    control;
  } uniform_data;
  static_assert(sizeof(uniform_data) == 32, "Uniform packing is wrong");

  public:
  Postprocess_Pass() {
    cs.reset();
    timer.init();
    uniform_buffer.reset();
    sampler.reset();
    input_image.reset();
    output_image.reset();
    feedback_buffer.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4)   (name color))
  (add_field (type u32)      (name control_flags))
)
#define CONTROL_ENABLE_FEEDBACK 1
bool is_control_set(uint bits) {
  return (control_flags & bits) != 0;
}
@(DECLARE_IMAGE
  (type READ_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name my_image)
)
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 2)
  (format RGBA32_FLOAT)
  (name out_image)
)
struct BVH_Node {
  float3 min;
  float3 max;
  u32    flags;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 3)
  (type BVH_Node)
  (name bvh_nodes)
)
struct Dummy {
  uint4    flags;
};
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 4)
  (type Dummy)
  (name dummy)
)
float4 op_laplace(int2 coords) {
  float4 val00 = image_load(my_image, coords + int2(-1, 0));
  float4 val01 = image_load(my_image, coords + int2(1, 0));
  float4 val10 = image_load(my_image, coords + int2(0, -1));
  float4 val11 = image_load(my_image, coords + int2(0, 1));
  float4 center = image_load(my_image, coords);
  float4 laplace = abs(center * 4.0 - val00 - val01 - val10 - val11) / 4.0;
  return laplace;
  // float intensity = dot(laplace, float4_splat(1.0)); 
  // return intensity > 0.5 ? float4_splat(1.0) : float4_splat(0.0);
}

@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(my_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  float2 uv = GLOBAL_THREAD_INDEX.xy / dim.xy;
  
  float4 in_val = image_load(my_image, GLOBAL_THREAD_INDEX.xy);
  in_val = pow(in_val, float4(1.0/2.2));
  if (GLOBAL_THREAD_INDEX.y == 777 && is_control_set(CONTROL_ENABLE_FEEDBACK)) {
    Dummy d;
    d.flags.x = GLOBAL_THREAD_INDEX.x;
    d.flags.y = GLOBAL_THREAD_INDEX.y;
    d.flags.z = GLOBAL_THREAD_INDEX.z;
    d.flags.w = 666;
    buffer_store(dummy, GLOBAL_THREAD_INDEX.x, d);
  }
  image_store(out_image, GLOBAL_THREAD_INDEX.xy, in_val);
@(END)
)"),
                                 NULL, 0);
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size       = sizeof(uniform_data);
      uniform_buffer      = pc->create_buffer(buf_info);
    }
    if (sampler.is_null()) {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      sampler             = pc->create_sampler(info);
    }
    input_image         = pc->get_resource(stref_s("opaque_pass/rt0"));
    rd::Image_Info info = pc->get_image_info(input_image);

    if (output_image.is_null() || width != info.width ||
        height != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      width  = info.width;
      height = info.height;
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
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      output_image = pc->create_image(info);
      pc->assign_name(output_image, stref_s("postprocess/img0"));
    }
    if (feedback_buffer.in_fly == false) {
      if (feedback_buffer.buffer.is_null()) {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits      = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits    = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
        buf_info.size          = 1 << 12;
        feedback_buffer.buffer = pc->create_buffer(buf_info);
      }
      feedback_buffer.fence = pc->get_fence(rd::Fence_Position::PASS_FINISED);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (input_image.is_null()) return;
    timer.update();
    ctx->CS_set_shader(cs);
    {
      Uniform *ptr = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->color   = float4(0.5f + 0.5f * std::cos(timer.cur_time), 0.0f, 0.0f,
                          0.5f + 0.5f * std::cos(timer.cur_time));
      ptr->control = 0;
      ptr->control |= feedback_buffer.in_fly ? 0 : 1;
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_rw_image(0, 1, 0, input_image, 0, 1, 0, 1);
    ctx->bind_rw_image(0, 2, 0, output_image, 0, 1, 0, 1);
    /*if (feedback_buffer.in_fly == false) {
      ctx->bind_storage_buffer(0, 4, feedback_buffer.buffer, 0, 1 << 12);
      feedback_buffer.in_fly = true;
    } else {
      if (ctx->get_fence_state(feedback_buffer.fence)) {
        u32 *ptr = (u32 *)ctx->map_buffer(feedback_buffer.buffer);
        fprintf(stdout, "feedback buffer is finished: %i %i %i %i ... \n",
                ptr[0], ptr[1], ptr[2], ptr[3]);
        ctx->unmap_buffer(feedback_buffer.buffer);
        feedback_buffer.in_fly = false;
      } else {
        fprintf(stdout, "feedback buffer is buisy\n");
      }
    }*/
    // ctx->bind_sampler(0, 2, sampler);
    ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
  }
  string_ref get_name() override { return stref_s("postprocess_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(feedback_buffer.buffer);
    timer.release();
  }
};

class Merge_Pass : public rd::IPass {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID uniform_buffer;
  Resource_ID sampler;
  Resource_ID my_image;
  Timer       timer;

  struct Uniform {
    float2x2 rot;
    float4   color;
  } uniform_data;

  public:
  Merge_Pass() {
    vs.reset();
    ps.reset();
    uniform_buffer.reset();
    sampler.reset();
    my_image.reset();
    timer.init();
  }
  void on_end(rd::IResource_Manager *rm) override {
    if (uniform_buffer.is_null() == false) {
      rm->release_resource(uniform_buffer);
      uniform_buffer.reset();
    }
  }
  void on_begin(rd::IResource_Manager *pc) override {
    rd::Image2D_Info info = pc->get_swapchain_image_info();
    width                 = info.width;
    height                = info.height;

    rd::Clear_Color cl;
    cl.clear = true;
    cl.r     = 0.0f;
    cl.g     = 0.0f;
    cl.b     = 0.0f;
    cl.a     = 0.0f;

    pc->add_render_target(pc->get_swapchain_image(), 0, 0, cl);
    static string_ref            shader    = stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4)   (name rot))
  (add_field (type float4)   (name color))
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name my_image)
)
@(DECLARE_SAMPLER
  (set 0)
  (binding 2)
  (name my_sampler)
)
#ifdef VERTEX
@(DECLARE_OUTPUT
  (location 0)
  (type float2)
  (name tex_coords)
)

@(ENTRY)
  float x = -1.0 + float((VERTEX_INDEX & 1) << 2);
  float y = -1.0 + float((VERTEX_INDEX & 2) << 1);
  float2 pos =float2(x, y);
  tex_coords = pos * 0.5 + float2(0.5, 0.5);
  @(EXPORT_POSITION
      float4(pos.xy, 0.5, 1.0)
  );
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT
  (location 0)
  (type float2)
  (name tex_coords)
)
@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  @(EXPORT_COLOR 0
    texture(sampler2D(my_image, my_sampler), tex_coords)
  );
@(END)
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);

    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
    buf_info.size       = sizeof(uniform_data);
    uniform_buffer      = pc->create_buffer(buf_info);
    if (sampler.is_null()) {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      sampler             = pc->create_sampler(info);
    }
    my_image = pc->get_resource(stref_s("postprocess/img0"));
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (my_image.is_null()) return;
    timer.update();
    setup_default_state(ctx);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    {
      Uniform *ptr = (Uniform *)ctx->map_buffer(uniform_buffer);
      ptr->rot     = float2x2(                                    //
          std::cos(timer.cur_time), std::sin(timer.cur_time), //
          -std::sin(timer.cur_time), std::cos(timer.cur_time) //
      );
      ptr->color   = float4(0.5f + 0.5f * std::cos(timer.cur_time), 0.0f, 0.0f,
                          0.5f + 0.5f * std::cos(timer.cur_time));
      ctx->unmap_buffer(uniform_buffer);
    }
    ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    ctx->bind_image(0, 1, 0, my_image, 0, 1, 0, 1);
    ctx->bind_sampler(0, 2, sampler);
    ctx->draw(3, 1, 0, 0);
  }
  string_ref get_name() override { return stref_s("simple_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
    timer.release();
  }
};

class GUI_Pass : public rd::IPass, public rd::IEvent_Consumer {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID sampler;
  Resource_ID vertex_buffer;
  Resource_ID index_buffer;

  Resource_ID font_texture;
  Resource_ID staging_buffer;

  Resource_ID opaque_rt0;

  unsigned char *font_pixels;
  int            font_width, font_height;

  i32         last_m_x;
  i32         last_m_y;
  ImDrawData *draw_data;
  Timer       timer;
  bool        imgui_initialized;

  public:
  void consume(void *_event) override {
    SDL_Event *event = (SDL_Event *)_event;
    if (imgui_initialized) {
      ImGui_ImplSDL2_ProcessEvent(event);
    }
    if (event->type == SDL_MOUSEMOTION) {
      SDL_MouseMotionEvent *m = (SDL_MouseMotionEvent *)event;
    }
  }
  GUI_Pass() {
    timer.init();
    imgui_initialized = false;
    draw_data         = NULL;
    last_m_x          = -1;
    last_m_y          = -1;
    font_texture.reset();
    staging_buffer.reset();
    vs.reset();
    ps.reset();
    vertex_buffer.reset();
    index_buffer.reset();
    sampler.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {
    rm->release_resource(vertex_buffer);
    rm->release_resource(index_buffer);
    if (staging_buffer.is_null() == false) {
      rm->release_resource(staging_buffer);
    }
    vertex_buffer.reset();
    index_buffer.reset();
    staging_buffer.reset();
  }
  void on_begin(rd::IResource_Manager *pc) override {
    rd::Image2D_Info info = pc->get_swapchain_image_info();
    width                 = info.width;
    height                = info.height;
    timer.update();
    rd::Clear_Color cl;
    MEMZERO(cl);
    cl.clear = true;

    pc->add_render_target(pc->get_swapchain_image(), 0, 0, cl);
    static string_ref            shader    = stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float2)   (name uScale))
  (add_field (type float2)   (name uTranslate))
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 0)
  (format RGBA32_FLOAT)
  (name sTexture)
)
@(DECLARE_SAMPLER
  (set 0)
  (binding 1)
  (name sSampler)
)
#ifdef VERTEX

@(DECLARE_INPUT (location 0) (type float2) (name aPos))
@(DECLARE_INPUT (location 1) (type float2) (name aUV))
@(DECLARE_INPUT (location 2) (type float4) (name aColor))

@(DECLARE_OUTPUT (location 0) (type float4) (name Color))
@(DECLARE_OUTPUT (location 1) (type float2) (name UV))

@(ENTRY)
  Color = aColor;
  UV = aUV;
  @(EXPORT_POSITION
      float4(aPos * uScale + uTranslate, 0, 1)
  );
@(END)
#endif
#ifdef PIXEL

@(DECLARE_INPUT (location 0) (type float4) (name Color))
@(DECLARE_INPUT (location 1) (type float2) (name UV))

@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  @(EXPORT_COLOR 0
    Color * texture(sampler2D(sTexture, sSampler), UV)
  );
@(END)
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);

    if (sampler.is_null()) {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      sampler             = pc->create_sampler(info);
    }

    if (!imgui_initialized) {
      imgui_initialized = true;
      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGuiIO &io = ImGui::GetIO();
      io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
      ImGui::StyleColorsDark();
      ImGui_ImplSDL2_InitForVulkan((SDL_Window *)pc->get_window_handle());

      io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_width, &font_height);

      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA8_UNORM;
      info.width      = font_width;
      info.height     = font_height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                        (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
      font_texture    = pc->create_image(info);
      io.Fonts->TexID = (ImTextureID)(intptr_t)font_texture.data;

      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      buf_info.size       = font_width * font_height * 4;
      staging_buffer      = pc->create_buffer(buf_info);
    }

    ImGui_ImplSDL2_NewFrame((SDL_Window *)pc->get_window_handle());
    ImGui::NewFrame();
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    window_flags |= ImGuiWindowFlags_NoBackground;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 1.0f);
    ImGui::SetNextWindowBgAlpha(-1.0f);
    ImGui::Begin("DockSpace", nullptr, window_flags);
    ImGui::PopStyleVar(4);
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();
    ImGuiIO &io = ImGui::GetIO();
    // static bool show_demo_window = true;
    // ImGui::ShowDemoWindow(&show_demo_window);
    {
      ImGui::Begin("Config");
      ImGui::Checkbox("enable_rasterization_pass",
                      &g_config.enable_rasterization_pass);
      ImGui::Checkbox("enable_compute_render_pass",
                      &g_config.enable_compute_render_pass);
      ImGui::End();
    }

    {
      ImGui::Begin("my window");
      auto  wpos        = ImGui::GetCursorScreenPos();
      auto  wsize       = ImGui::GetWindowSize();
      float height_diff = 24;
      if (wsize.y < height_diff + 2) {
        wsize.y = 2;
      } else {
        wsize.y = wsize.y - height_diff;
      }
      ((Opaque_Pass *)pc->get_pass(stref_s("opaque_pass")))
          ->set_size(wsize.x, wsize.y);

      if (ImGui::IsWindowHovered()) {
        f32 camera_speed = 2.0f;
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_LSHIFT]) {
          camera_speed = 10.0f;
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
      opaque_rt0 = pc->get_resource(stref_s("opaque_pass/rt0"));
      ImGui::Image((ImTextureID)(intptr_t)opaque_rt0.data,
                   ImVec2(wsize.x, wsize.y));
      ImGui::End();
    }
    ImGui::Render();

    draw_data = ImGui::GetDrawData();
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size       = (draw_data->TotalVtxCount + 1) * sizeof(ImDrawVert);
      vertex_buffer       = pc->create_buffer(buf_info);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size       = (draw_data->TotalIdxCount + 1) * sizeof(ImDrawIdx);
      index_buffer        = pc->create_buffer(buf_info);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    {
      ImDrawVert *vtx_dst = (ImDrawVert *)ctx->map_buffer(vertex_buffer);
      ImDrawIdx * idx_dst = (ImDrawIdx *)ctx->map_buffer(index_buffer);
      ito(draw_data->CmdListsCount) {

        const ImDrawList *cmd_list = draw_data->CmdLists[i];
        memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
               cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idx_dst, cmd_list->IdxBuffer.Data,
               cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_dst += cmd_list->VtxBuffer.Size;
        idx_dst += cmd_list->IdxBuffer.Size;
      }
      ctx->unmap_buffer(vertex_buffer);
      ctx->unmap_buffer(index_buffer);
    }
    if (font_pixels != NULL) {
      void *dst = ctx->map_buffer(staging_buffer);
      memcpy(dst, font_pixels, font_width * font_height * 4);
      ctx->unmap_buffer(staging_buffer);
      ctx->copy_buffer_to_image(staging_buffer, 0, font_texture, 0, 0);
      font_pixels = NULL;
    }
    setup_default_state(ctx);
    rd::Blend_State bs;
    MEMZERO(bs);
    bs.enabled          = true;
    bs.alpha_blend_op   = rd::Blend_OP::ADD;
    bs.color_blend_op   = rd::Blend_OP::ADD;
    bs.dst_alpha        = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
    bs.src_alpha        = rd::Blend_Factor::SRC_ALPHA;
    bs.dst_color        = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
    bs.src_color        = rd::Blend_Factor::SRC_ALPHA;
    bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT |
                          (u32)rd::Color_Component_Bit::G_BIT |
                          (u32)rd::Color_Component_Bit::B_BIT |
                          (u32)rd::Color_Component_Bit::A_BIT;
    ctx->OM_set_blend_state(0, bs);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->bind_sampler(0, 1, sampler);
    ImVec2 clip_off          = draw_data->DisplayPos;
    ImVec2 clip_scale        = draw_data->FramebufferScale;
    int    global_vtx_offset = 0;
    int    global_idx_offset = 0;
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 1;
      info.offset   = 8;
      info.type     = rd::Attriute_t::TEXCOORD0;
      ctx->IA_set_attribute(info);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGBA8_UNORM;
      info.location = 2;
      info.offset   = 16;
      info.type     = rd::Attriute_t::TEXCOORD1;
      ctx->IA_set_attribute(info);
    }
    {
      float scale[2];
      scale[0] = 2.0f / draw_data->DisplaySize.x;
      scale[1] = 2.0f / draw_data->DisplaySize.y;
      float translate[2];
      translate[0] = -1.0f - draw_data->DisplayPos.x * scale[0];
      translate[1] = -1.0f - draw_data->DisplayPos.y * scale[1];
      ctx->push_constants(scale, 0, 8);
      ctx->push_constants(translate, 8, 8);
    }
    ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT16);
    ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, sizeof(ImDrawVert),
                              rd::Input_Rate::VERTEX);
    for (int n = 0; n < draw_data->CmdListsCount; n++) {
      const ImDrawList *cmd_list = draw_data->CmdLists[n];
      for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
        const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[cmd_i];
        ImVec4           clip_rect;
        clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
        clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
        clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
        clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;
        Resource_ID res_id;
        res_id.data = (u64)pcmd->TextureId;
        ctx->bind_image(0, 0, 0, res_id, 0, 1, 0, 1);
        ctx->set_scissor(clip_rect.x, clip_rect.y, clip_rect.z - clip_rect.x,
                         clip_rect.w - clip_rect.y);
        ctx->draw_indexed(pcmd->ElemCount, 1,
                          pcmd->IdxOffset + global_idx_offset, 0,
                          pcmd->VtxOffset + global_vtx_offset);
      }
      global_idx_offset += cmd_list->IdxBuffer.Size;
      global_vtx_offset += cmd_list->VtxBuffer.Size;
    }

    ImGuiIO &io = ImGui::GetIO();
  }
  string_ref get_name() override { return stref_s("simple_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    rm->release_resource(vs);
    rm->release_resource(ps);
    delete this;
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  g_scene.init();
  GUI_Pass *    gui  = new GUI_Pass;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  pmng->set_event_consumer(gui);
  // pmng->add_pass(rd::Pass_t::RENDER, new Mesh_Prepare_Pass);
  // pmng->add_pass(rd::Pass_t::COMPUTE, new Culling_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, new Opaque_Pass);
  // pmng->add_pass(rd::Pass_t::COMPUTE, new Compute_Render_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Postprocess_Pass);
  // pmng->add_pass(rd::Pass_t::RENDER, new Merge_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, gui);
  pmng->loop();
  return 0;
}
