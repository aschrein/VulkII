#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "script.hpp"

#include "scene.hpp"

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

#include <imgui.h>
#include <imgui/examples/imgui_impl_sdl.h>

Config g_config;
Camera g_camera;

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
(config
 (add bool enable_rasterization_pass 1)
 (add bool enable_compute_depth 1)
 (add bool enable_compute_render_pass 1)
 (add u32 triangles_per_lane 1)
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
  InlineArray<size_t, 16>    meshlet_attribute_offsets;
  InlineArray<size_t, 16>    attribute_sizes;
  InlineArray<size_t, 16>    meshlet_attribute_sizes;
  InlineArray<Attribute, 16> attributes;
  size_t                     total_memory_needed;
  u32                        total_indices;
  size_t                     index_offset;
  size_t                     meshlet_index_offset;
  size_t                     meshlet_total_index_data;
  size_t                     meshlet_data_offset;
  size_t                     meshlet_data_size;
  void                       init(string_ref name) {
    MeshNode::init(name);
    total_memory_needed = 0;
    total_indices       = 0;
    index_offset        = 0;
    attribute_offsets.init();
    meshlet_attribute_offsets.init();
    attribute_sizes.init();
    meshlet_attribute_sizes.init();
    attributes.init();
  }

  struct GPU_Meshlet {
    u32 vertex_offset;
    u32 index_offset;
    u32 triangle_count;
    u32 vertex_count;
    // (x, y, z, radius)
    float4 sphere;
    // (x, y, z, _)
    float4 cone_apex;
    // (x, y, z, cutoff)
    float4 cone_axis_cutoff;
    u32    cone_pack;
  };
  static_assert(sizeof(GPU_Meshlet) == 68, "Packing error");

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
        attribute_sizes[i] += rd::IResource_Manager::BUFFER_ALIGNMENT;
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
      // Calculate meshlets
      meshlet_attribute_sizes.resize(attributes.size);
      meshlet_attribute_offsets.resize(attributes.size);
      total_memory_needed =
          rd::IResource_Manager::align_up(total_memory_needed);
      u32 meshlet_vertex_offset = total_memory_needed;
      ito(primitives.size) {
        Raw_Meshlets_Opaque meshlets = primitives[i].meshlets;
        jto(meshlets.attributes.size) {
          meshlet_attribute_sizes[j] += meshlets.get_attribute_size(j);
        }
      }
      ito(attributes.size) {
        meshlet_attribute_sizes[i] += rd::IResource_Manager::BUFFER_ALIGNMENT;
        jto(i) { meshlet_attribute_offsets[i] += meshlet_attribute_sizes[j]; }
        meshlet_attribute_offsets[i] =
            meshlet_vertex_offset +
            rd::IResource_Manager::align_up(meshlet_attribute_offsets[i]);
        total_memory_needed =
            meshlet_attribute_offsets[i] + meshlet_attribute_sizes[i];
      }
      total_memory_needed =
          rd::IResource_Manager::align_up(total_memory_needed);
      meshlet_index_offset     = total_memory_needed;
      meshlet_total_index_data = 0;
      ito(primitives.size) {
        Raw_Meshlets_Opaque meshlets = primitives[i].meshlets;
        total_memory_needed += meshlets.index_data.size;
        meshlet_total_index_data += meshlets.index_data.size;
      }
      total_memory_needed =
          rd::IResource_Manager::align_up(total_memory_needed);
      meshlet_data_offset = total_memory_needed;
      meshlet_data_size   = 0;
      ito(primitives.size) {
        Raw_Meshlets_Opaque meshlets = primitives[i].meshlets;
        meshlet_data_size += meshlets.meshlets.size * sizeof(GPU_Meshlet);
        total_memory_needed += meshlets.meshlets.size * sizeof(GPU_Meshlet);
      }
    }
    return total_memory_needed;
  }
  void put_data(void *ptr) {
    // put vertex data
    {
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
    // put meshlets
    {
      InlineArray<size_t, 16> attribute_cursors;
      MEMZERO(attribute_cursors);
      size_t indices_offset = 0;
      size_t meshlet_offset = 0;
      ito(primitives.size) {
        Raw_Meshlets_Opaque meshlets = primitives[i].meshlets;
        jto(meshlets.attributes.size) {
          Attribute attribute      = meshlets.attributes[j];
          size_t    attribute_size = meshlets.get_attribute_size(j);
          memcpy(
              (u8 *)ptr + meshlet_attribute_offsets[j] + attribute_cursors[j],
              &meshlets.attribute_data[0] + attribute.offset, attribute_size);
          attribute_cursors[j] += attribute_size;
        }
        size_t index_size = meshlets.index_data.size;
        memcpy((u8 *)ptr + meshlet_index_offset + indices_offset,
               &meshlets.index_data[0], index_size);
        indices_offset += index_size;
        jto(meshlets.meshlets.size) {
          GPU_Meshlet d;
          Meshlet     m = meshlets.meshlets[j];
          static_assert(sizeof(d) == sizeof(m), "Packing error");
          memcpy(&d, &m, sizeof(m));
          memcpy((u8 *)ptr + meshlet_data_offset + meshlet_offset, &d,
                 sizeof(GPU_Meshlet));
          meshlet_offset += sizeof(GPU_Meshlet);
        }
      }
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
  void dispatch(rd::Imm_Ctx *ctx, Resource_ID vertex_buffer, size_t offset,
                u32 push_offset) {
    static u32 attribute_to_location[] = {
        0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    ito(attributes.size) {
      Attribute attr = attributes[i];
      ctx->bind_storage_buffer(2, 1 + attribute_to_location[(u32)attr.type],
                               vertex_buffer, offset + attribute_offsets[i],
                               attribute_sizes[i]);
    }
    ctx->bind_storage_buffer(2, 0, vertex_buffer, offset + index_offset,
                             get_num_indices() * 4);
    u32 vertex_cursor = 0;
    u32 index_cursor  = 0;
    ito(primitives.size) {
      ctx->push_constants(&primitives[i].mesh.num_indices, push_offset, 4);
      ctx->push_constants(&index_cursor, push_offset + 4, 4);
      ctx->push_constants(&vertex_cursor, push_offset + 8, 4);
      ctx->dispatch(((primitives[i].mesh.num_indices + 255) / 256 +
                     g_config.get_u32("triangles_per_lane") - 1) /
                        g_config.get_u32("triangles_per_lane"),
                    1, 1);
      index_cursor += primitives[i].mesh.num_indices;
      vertex_cursor += primitives[i].mesh.num_vertices;
    }
  }
  void dispatch_meshlets(rd::Imm_Ctx *ctx, Resource_ID vertex_buffer,
                         size_t offset, u32 push_offset) {
    static u32 attribute_to_location[] = {
        0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    ito(attributes.size) {
      Attribute attr = attributes[i];
      ctx->bind_storage_buffer(
          2, 1 + attribute_to_location[(u32)attr.type], vertex_buffer,
          offset + meshlet_attribute_offsets[i], meshlet_attribute_sizes[i]);
    }
    ctx->bind_storage_buffer(2, 0, vertex_buffer, offset + meshlet_index_offset,
                             meshlet_total_index_data);
    ctx->bind_storage_buffer(1, 0, vertex_buffer, offset + meshlet_data_offset,
                             meshlet_data_size);
    u32 vertex_cursor  = 0;
    u32 index_cursor   = 0;
    u32 meshlet_cursor = 0;
    u32 max_num_groups = 1024;
    ito(primitives.size) {
      Raw_Meshlets_Opaque &meshlets = primitives[i].meshlets;
      jto(((meshlets.meshlets.size + max_num_groups - 1) / max_num_groups)) {
        u32 first_meshlet = j * max_num_groups;
        u32 num_meshlets =
            MIN(meshlets.meshlets.size - j * max_num_groups, max_num_groups);
        ctx->push_constants(&num_meshlets, push_offset, 4);
        ctx->push_constants(&meshlet_cursor, push_offset + 4, 4);
        ctx->push_constants(&index_cursor, push_offset + 8, 4);
        ctx->push_constants(&vertex_cursor, push_offset + 12, 4);

        ctx->dispatch(num_meshlets, 1, 1);

        meshlet_cursor += num_meshlets;
      }
      u32 num_vertices = 0;
      u32 num_indices  = 0;
      kto(meshlets.meshlets.size) {
        num_vertices += meshlets.meshlets[k].vertex_count;
        num_indices += meshlets.meshlets[k].triangle_count * 3;
      }
      index_cursor += num_indices;
      vertex_cursor += num_vertices;
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
  Array<size_t>        meshlet_offsets;
  bool                 gfx_buffers_initialized;
  u32                  texture_streaming_id;
  size_t               total_memory;
  Resource_ID          vertex_buffer;
  Resource_ID          staging_vertex_buffer;
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
    total_memory            = 0;
    dummy_initialized       = false;
    gfx_buffers_initialized = false;
    texture_streaming_id    = 0;
    meshes.init();
    vertex_buffer.reset();
    staging_vertex_buffer.reset();
    dummy_texture.reset();
    staging_buffer.reset();
    gfx_images.init();
    images.init();
    SceneFactory sf(this);
    root = Node::create(stref_s("ROOT"));
    root->add_child(load_gltf_pbr(&sf, stref_s("models/light/scene_low.gltf"))
                        ->rename(stref_s("LOW")));
    root->add_child(load_gltf_pbr(&sf, stref_s("models/light/scene.gltf"))
                        ->rename(stref_s("HIGH")));
    // root = load_gltf_pbr(
    //    &sf, stref_s("models/chateau-de-marvao-portugal/scene.gltf"));
    root->dump();
  }

  Node *get_node(string_ref name) {
    Node *out = NULL;
    traverse([&](Node *node) {
      if (node->get_name() == name) {
        out = node;
      }
    });
    return out;
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
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                              (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER |
                              (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST |
                              (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
        buf_info.size = total_memory;
        vertex_buffer = rm->create_buffer(buf_info);
      }
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits     = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits   = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
        buf_info.size         = total_memory;
        staging_vertex_buffer = rm->create_buffer(buf_info);
      }
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
    if (staging_buffer.is_null() == false) {
      rm->release_resource(staging_buffer);
      staging_buffer.reset();
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
  void gfx_bind(rd::Imm_Ctx *ctx) {
    if (dummy_initialized == false) {
      {
        u32 *ptr = (u32 *)ctx->map_buffer(staging_buffer);
        ito(16 * 16) { ptr[i] = 0xff0000ffu; }
        ctx->unmap_buffer(staging_buffer);
        ctx->copy_buffer_to_image(staging_buffer, 0, dummy_texture, 0, 0);
      }
      {
        u8 *ptr = (u8 *)ctx->map_buffer(staging_vertex_buffer);
        ito(meshes.size) { meshes[i]->put_data(ptr + mesh_offsets[i]); }
        ctx->unmap_buffer(staging_vertex_buffer);
        ctx->copy_buffer(staging_vertex_buffer, 0, vertex_buffer, 0,
                         total_memory);
      }
      dummy_initialized = true;
    } else if (texture_streaming_id < gfx_images.size) {
      /* void *ptr = ctx->map_buffer(staging_buffer);
       memcpy(ptr, images[texture_streaming_id].data,
              images[texture_streaming_id].get_size_in_bytes());
       ctx->unmap_buffer(staging_buffer);
       ctx->copy_buffer_to_image(staging_buffer, 0,
                                 gfx_images[texture_streaming_id], 0, 0);
       texture_streaming_id++;*/
    }
    ito(gfx_images.size) {
      if (i < texture_streaming_id) {
        ctx->bind_image(1, 1, i, gfx_images[i], 0, 1, 0, 1);
      } else {
        ctx->bind_image(1, 1, i, dummy_texture, 0, 1, 0, 1);
      }
    }
  }
  void gfx_exec(rd::Imm_Ctx *ctx) {
    gfx_bind(ctx);
    ito(meshes.size) { meshes[i]->draw(ctx, vertex_buffer, mesh_offsets[i]); }
  }
  void gfx_dispatch(rd::Imm_Ctx *ctx, u32 push_offset) {
    gfx_bind(ctx);
    ito(meshes.size) {
      meshes[i]->dispatch(ctx, vertex_buffer, mesh_offsets[i], push_offset);
    }
  }
  void gfx_dispatch_meshlets(rd::Imm_Ctx *ctx, u32 push_offset) {
    gfx_bind(ctx);
    ito(meshes.size) {
      meshes[i]->dispatch_meshlets(ctx, vertex_buffer, mesh_offsets[i],
                                   push_offset);
    }
  }

  void on_pass_end(rd::IResource_Manager *rm) {
    if (staging_vertex_buffer.is_null() == false) {
      rm->release_resource(staging_vertex_buffer);
      staging_vertex_buffer.reset();
    }
  }
  void release_gfx(rd::IResource_Manager *rm) {
    ito(gfx_images.size) rm->release_resource(gfx_images[i]);
    gfx_images.release();
    rm->release_resource(vertex_buffer);
    rm->release_resource(staging_vertex_buffer);
    if (staging_buffer.is_null() == false) {
      rm->release_resource(staging_buffer);
      staging_buffer.reset();
    }
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
  void on_begin(rd::IResource_Manager *pc) override {
    // rd::Image2D_Info info = pc->get_swapchain_image_info();
    //    width                 = info.width;
    // height                = info.height;
    g_camera.aspect = (float)g_config.get_u32("g_buffer_width") /
                      g_config.get_u32("g_buffer_height");
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
      rt0_info.width      = g_config.get_u32("g_buffer_width");
      rt0_info.height     = g_config.get_u32("g_buffer_height");
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
      info.width      = g_config.get_u32("g_buffer_width");
      info.height     = g_config.get_u32("g_buffer_height");
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

//@(DECLARE_INPUT (location 2) (type float3) (name BINORMAL))
//@(DECLARE_INPUT (location 3) (type float3) (name TANGENT))
//@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))
//@(DECLARE_INPUT (location 5) (type float2) (name TEXCOORD1))
//@(DECLARE_INPUT (location 6) (type float2) (name TEXCOORD2))
//@(DECLARE_INPUT (location 7) (type float2) (name TEXCOORD3))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))

//@(DECLARE_OUTPUT (location 2) (type float3) (name PIXEL_BINORMAL))
//@(DECLARE_OUTPUT (location 3) (type float3) (name PIXEL_TANGENT))
//@(DECLARE_OUTPUT (location 4) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_OUTPUT (location 5) (type uint) (name PIXEL_INSTANCE_ID))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_NORMAL    = NORMAL;
  //PIXEL_BINORMAL  = BINORMAL;
  //PIXEL_TANGENT   = TANGENT;
  //PIXEL_TEXCOORD0 = TEXCOORD0;
  PIXEL_INSTANCE_ID = INSTANCE_INDEX;
  float3 position = POSITION;
  // float4x4 world_matrix = buffer_load(instance_infos, INSTANCE_INDEX).model;
  @(EXPORT_POSITION
      viewproj * float4(position, 1.0)
  );
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
//@(DECLARE_INPUT (location 2) (type float3) (name PIXEL_BINORMAL))
//@(DECLARE_INPUT (location 3) (type float3) (name PIXEL_TANGENT))
//@(DECLARE_INPUT (location 4) (type float3) (name PIXEL_TEXCOORD0))
@(DECLARE_INPUT (location 5) (type "flat uint") (name PIXEL_INSTANCE_ID))

@(DECLARE_RENDER_TARGET
  (location 0)
)
@(ENTRY)
  float4 albedo = float4(0.0, 1.0, 1.0, 1.0);
  float4 color = float4_splat(1.0) * (0.5 + 0.5 * dot(PIXEL_NORMAL.rgb, normalize(float3(1.0, 1.0, 1.0))));
  /*i32 albedo_id = buffer_load(instance_infos, instance_index).albedo_id;
  if (albedo_id >= 0) {
    albedo = texture(sampler2D(material_textures[nonuniformEXT(albedo_id)], my_sampler), tex_coords);
  }*/
  @(EXPORT_COLOR 0 color);
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
    if (g_config.get_bool("enable_rasterization_pass") == false) return;
    setup_default_state(ctx);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)g_config.get_u32("g_buffer_width"),
                      (float)g_config.get_u32("g_buffer_height"), 0.0f, 1.0f);
    ctx->set_scissor(0, 0, g_config.get_u32("g_buffer_width"),
                     g_config.get_u32("g_buffer_height"));
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    rs_state.line_width   = 1.0f;
    rs_state.depth_bias   = 0.0f;
    ctx->RS_set_state(rs_state);
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
  Resource_ID clear_cs;
  Resource_ID output_image;
  Resource_ID output_depth;

  public:
  Compute_Render_Pass() {
    cs.reset();
    clear_cs.reset();
    output_image.reset();
    output_depth.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {}
  void on_begin(rd::IResource_Manager *pc) override {
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type u32)       (name control_flags))
  (add_field (type u32)       (name num_triangles_per_lane))
  (add_field (type u32)       (name index_count))
  (add_field (type u32)       (name first_index))
  (add_field (type i32)       (name vertex_offset))
)

#define CONTROL_DEPTH_ENABLE 1
#define is_control(flag) (control_flags & flag) != 0

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
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 0)
  (type u32)
  (name index_buffer)
)

struct Vertex {
  float3 position;
  float3 normal;
};

#define FETCH_FLOAT3

#ifdef FETCH_FLOAT3
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float3)
  (name position_buffer)
)
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float3)
  (name normal_buffer)
)
Vertex fetch(u32 index) {
  Vertex o;  
  o.position = buffer_load(position_buffer, index);
  o.normal = buffer_load(normal_buffer, index);
  return o;
}
#else
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float)
  (name position_buffer)
)
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float)
  (name normal_buffer)
)

Vertex fetch(u32 index) {
  Vertex o;  
  o.position.x = buffer_load(position_buffer, index * 3 + 0);
  o.position.y = buffer_load(position_buffer, index * 3 + 1);
  o.position.z = buffer_load(position_buffer, index * 3 + 2);
  o.normal.x = buffer_load(normal_buffer, index * 3 + 0);
  o.normal.y = buffer_load(normal_buffer, index * 3 + 1);
  o.normal.z = buffer_load(normal_buffer, index * 3 + 2);
  return o;
}
#endif

@(GROUP_SIZE 256 1 1)
@(ENTRY)
  for (u32 iter_id = 0; iter_id < num_triangles_per_lane; iter_id++) {
    u32 triangle_index = iter_id + GLOBAL_THREAD_INDEX.x * num_triangles_per_lane;
    if (triangle_index > index_count / 3)
      return;

    u32 i0 = buffer_load(index_buffer, first_index + triangle_index * 3 + 0);
    u32 i1 = buffer_load(index_buffer, first_index + triangle_index * 3 + 1);
    u32 i2 = buffer_load(index_buffer, first_index + triangle_index * 3 + 2);
   
    Vertex v0 = fetch(vertex_offset + i0);
    Vertex v1 = fetch(vertex_offset + i1);
    Vertex v2 = fetch(vertex_offset + i2);

    float4 pp0 = mul4(viewproj, float4(v0.position, 1.0));
    float4 pp1 = mul4(viewproj, float4(v1.position, 1.0));
    float4 pp2 = mul4(viewproj, float4(v2.position, 1.0));
    pp0.xyz /= pp0.w;
    pp1.xyz /= pp1.w;
    pp2.xyz /= pp2.w;
    {
      float2 e0 = pp0.xy - pp2.xy; 
      float2 e1 = pp1.xy - pp0.xy; 
      float2 e2 = pp2.xy - pp1.xy; 
      float2 n0 = float2(e0.y, -e0.x);
      if (dot(e1, n0) > 0.0)
        continue;
    }
    float area = 1.0;
    float b0 = 1.0 / 3.0;
    float b1 = 1.0 / 3.0;
    float b2 = 1.0 / 3.0;
    b0 /= area;
    b1 /= area;
    b2 /= area;
    float z = 1.0 / (b0 / pp0.w + b1 / pp1.w + b2 / pp2.w);

    float3 pp = pp0.xyz * b0 + pp1.xyz * b1 + pp2.xyz * b2;
    float3 n = normalize(z * (v0.normal * b0 + v1.normal * b1 + v1.normal * b2));
    int2 dim = imageSize(out_image);
    if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
      continue;
    i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
    i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
    if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
      float4 color = float4_splat(1.0) * (0.5 + 0.5 * dot(n, normalize(float3(1.0, 1.0, 1.0))));
      if (is_control(CONTROL_DEPTH_ENABLE)) {
        u32 depth = u32(1.0 / pp.z);
        if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
          image_store(out_image, int2(x, y), color);
        }
      } else {
        image_store(out_image, int2(x, y), color);
      }
    }
  }
@(END)
)"),
                                 NULL, 0);
    if (clear_cs.is_null())
      clear_cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
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
@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  image_store(out_image, int2(GLOBAL_THREAD_INDEX.xy), float4(0.0, 0.0, 0.0, 1.0));
  image_store(out_depth, int2(GLOBAL_THREAD_INDEX.xy), uint4(1 << 31, 0, 0, 0));
@(END)
)"),
                                       NULL, 0);
    rd::Image_Info info;
    MEMZERO(info);
    if (output_image.is_null() == false)
      info = pc->get_image_info(output_image);
    if (output_image.is_null() ||
        g_config.get_u32("g_buffer_width") != info.width ||
        g_config.get_u32("g_buffer_height") != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      if (output_depth.is_null() == false) pc->release_resource(output_depth);
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::RGBA32_FLOAT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_image = pc->create_image(info);
      }
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::R32_UINT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_depth = pc->create_image(info);
      }
      pc->assign_name(output_image, stref_s("compute_render/img0"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.get_bool("enable_compute_render_pass") == false) return;
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->CS_set_shader(clear_cs);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth,
                       (u32)rd::Access_Bits::SHADER_WRITE |
                           (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 0, 0, output_image, 0, 1, 0, 1);
    ctx->bind_rw_image(0, 1, 0, output_depth, 0, 1, 0, 1);
    ctx->dispatch((g_config.get_u32("g_buffer_width") + 15) / 16,
                  (g_config.get_u32("g_buffer_height") + 15) / 16, 1);
    ctx->CS_set_shader(cs);
    float4x4 vp = g_camera.viewproj();
    ctx->push_constants(&vp, 0, sizeof(vp));
    u32 control_flags = 0;
    control_flags |= (g_config.get_bool("enable_compute_depth") ? 1 : 0) << 0;
    ctx->push_constants(&control_flags, 64, 4);
    g_config.get_u32("triangles_per_lane") =
        MAX(0, MIN(128, g_config.get_u32("triangles_per_lane")));
    ctx->push_constants(&g_config.get_u32("triangles_per_lane"), 64 + 4, 4);
    g_scene.gfx_dispatch(ctx, 72);
  }
  string_ref get_name() override { return stref_s("compute_render_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(output_image);
  }
};

class Meshlet_Tile {
  Resource_ID cs;
  Resource_ID clear_cs;
  Resource_ID output_image;
  Resource_ID output_depth;
  u32         width;
  u32         height;

  public:
  void init(u32 width, u32 height) {
    cs.reset();
    clear_cs.reset();
    output_image.reset();
    output_depth.reset();
    this->width  = width;
    this->height = height;
  }
  void on_end(rd::IResource_Manager *rm) {}
  void on_begin(rd::IResource_Manager *pc) {
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type u32)       (name num_meshlets))
  (add_field (type u32)       (name meshlet_offset))
  (add_field (type u32)       (name index_offset))
  (add_field (type u32)       (name vertex_offset))
)

#define CONTROL_DEPTH_ENABLE 1
#define is_control(flag) (control_flags & flag) != 0

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
struct Meshlet {
  u32 vertex_offset;
  u32 index_offset;
  u32  triangle_count;
  u32  vertex_count;
  float4 sphere;
  float4 cone_apex;
  float4 cone_axis_cutoff;
  u32 cone_pack;
};

struct Vertex {
  float3 position;
  float3 normal;
};

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 1)
  (binding 0)
  (type Meshlet)
  (name meshlet_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 0)
  (type u32)
  (name index_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float3)
  (name position_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float3)
  (name normal_buffer)
)

Vertex fetch(u32 index) {
  Vertex o;  
  o.position = buffer_load(position_buffer, index);
  o.normal = buffer_load(normal_buffer, index);
  return o;
}

u32 fetch_index(u32 index) {
  u32 raw = buffer_load(index_buffer, (index_offset + index) / 4);
  u32 sub_index = index & 0x3;
  return (raw >> (sub_index * 8)) & 0xffu;
}

// shared Vertex gs_vertices[64];

@(GROUP_SIZE 256 1 1)
@(ENTRY)
  
  u32 meshlet_index = GROUPT_INDEX.x;
  if (meshlet_index > num_meshlets)
    return;

  Meshlet meshlet = buffer_load(meshlet_buffer, meshlet_offset + meshlet_index);
  
  {
    float4 cp = mul4(viewproj, float4(meshlet.cone_apex.xyz, 1.0));
    float4 ca = mul4(viewproj, float4(meshlet.cone_apex.xyz + meshlet.cone_axis_cutoff.xyz, 1.0));
    if (ca.z > cp.z)
      return;
  }
  //if (LOCAL_THREAD_INDEX.x < 64)
  //  gs_vertices[LOCAL_THREAD_INDEX.x] = fetch(vertex_offset + meshlet.vertex_offset + LOCAL_THREAD_INDEX.x);

  // barrier();

  u32 triangle_index = LOCAL_THREAD_INDEX.x;

  if (triangle_index > meshlet.triangle_count)
    return;

  u32 i0 = fetch_index(meshlet.index_offset + triangle_index * 3 + 0);
  u32 i1 = fetch_index(meshlet.index_offset + triangle_index * 3 + 1);
  u32 i2 = fetch_index(meshlet.index_offset + triangle_index * 3 + 2);
   
  //Vertex v0 = gs_vertices[i0];//fetch(vertex_offset + meshlet.vertex_offset + i0);
  Vertex v0 = fetch(vertex_offset + meshlet.vertex_offset + i0);
  //Vertex v1 = gs_vertices[i1];//fetch(vertex_offset + meshlet.vertex_offset + i1);
  Vertex v1 = fetch(vertex_offset + meshlet.vertex_offset + i1);
  //Vertex v2 = gs_vertices[i2];//fetch(vertex_offset + meshlet.vertex_offset + i2);
  Vertex v2 = fetch(vertex_offset + meshlet.vertex_offset + i2);

  float4 pp0 = mul4(viewproj, float4(v0.position, 1.0));
  float4 pp1 = mul4(viewproj, float4(v1.position, 1.0));
  float4 pp2 = mul4(viewproj, float4(v2.position, 1.0));
  pp0.xyz /= pp0.w;
  pp1.xyz /= pp1.w;
  pp2.xyz /= pp2.w;
  {
    float2 e0 = pp0.xy - pp2.xy; 
    float2 e1 = pp1.xy - pp0.xy; 
    float2 e2 = pp2.xy - pp1.xy; 
    float2 n0 = float2(e0.y, -e0.x);
    if (dot(e1, n0) > 0.0)
      return;
  }
  float area = 1.0;
  float b0 = 1.0 / 3.0;
  float b1 = 1.0 / 3.0;
  float b2 = 1.0 / 3.0;
  b0 /= area;
  b1 /= area;
  b2 /= area;
  float z = 1.0 / (b0 / pp0.w + b1 / pp1.w + b2 / pp2.w);

  float3 pp = pp0.xyz * b0 + pp1.xyz * b1 + pp2.xyz * b2;
  float3 n = normalize(z * (v0.normal * b0 + v1.normal * b1 + v1.normal * b2));
  int2 dim = imageSize(out_image);
  if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
    return;
  i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
  i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
  if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
    float4 color = float4(
        float(((31 * meshlet_index) >> 0) % 255) / 255.0,
        float(((31 * meshlet_index) >> 8) % 255) / 255.0,
        float(((31 * meshlet_index) >> 16) % 255) / 255.0,
        1.0
        );
    // color = float4((0.5 * n + float3_splat(0.5)), 1.0);
    u32 depth = u32(1.0 / pp.z);
    if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
      image_store(out_image, int2(x, y), color);
    }
  }
@(END)
)"),
                                 NULL, 0);
    if (clear_cs.is_null())
      clear_cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
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
@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  image_store(out_image, int2(GLOBAL_THREAD_INDEX.xy), float4(0.0, 0.0, 0.0, 1.0));
  image_store(out_depth, int2(GLOBAL_THREAD_INDEX.xy), uint4(1 << 31, 0, 0, 0));
@(END)
)"),
                                       NULL, 0);
    rd::Image_Info info;
    MEMZERO(info);
    if (output_image.is_null() == false)
      info = pc->get_image_info(output_image);
    if (output_image.is_null() ||
        g_config.get_u32("g_buffer_width") != info.width ||
        g_config.get_u32("g_buffer_height") != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      if (output_depth.is_null() == false) pc->release_resource(output_depth);
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::RGBA32_FLOAT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_image = pc->create_image(info);
      }
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::R32_UINT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_depth = pc->create_image(info);
      }
      pc->assign_name(output_image, stref_s("meshlet_render/img0"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) {
    if (g_config.get_bool("enable_meshlets_render_pass") == false) return;
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->CS_set_shader(clear_cs);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth,
                       (u32)rd::Access_Bits::SHADER_WRITE |
                           (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 0, 0, output_image, 0, 1, 0, 1);
    ctx->bind_rw_image(0, 1, 0, output_depth, 0, 1, 0, 1);
    ctx->dispatch((g_config.get_u32("g_buffer_width") + 15) / 16,
                  (g_config.get_u32("g_buffer_height") + 15) / 16, 1);
    ctx->CS_set_shader(cs);
    float4x4 vp = g_camera.viewproj();
    ctx->push_constants(&vp, 0, sizeof(vp));
    u32 control_flags = 0;
    control_flags |= (g_config.get_bool("enable_compute_depth") ? 1 : 0) << 0;
    ctx->push_constants(&control_flags, 64, 4);
    g_scene.gfx_dispatch_meshlets(ctx, 68);
  }
  void release(rd::IResource_Manager *rm) {
    rm->release_resource(cs);
    rm->release_resource(clear_cs);
    rm->release_resource(output_image);
    rm->release_resource(output_depth);
  }
};

class Meshlet_Compute_Render_Pass : public rd::IPass {
  Resource_ID cs;
  Resource_ID clear_cs;
  Resource_ID output_image;
  Resource_ID output_depth;

  public:
  Meshlet_Compute_Render_Pass() {
    cs.reset();
    clear_cs.reset();
    output_image.reset();
    output_depth.reset();
  }
  void on_end(rd::IResource_Manager *rm) override {}
  void on_begin(rd::IResource_Manager *pc) override {
    if (cs.is_null())
      cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type u32)       (name num_meshlets))
  (add_field (type u32)       (name meshlet_offset))
  (add_field (type u32)       (name index_offset))
  (add_field (type u32)       (name vertex_offset))
)

#define CONTROL_DEPTH_ENABLE 1
#define is_control(flag) (control_flags & flag) != 0

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
struct Meshlet {
  u32 vertex_offset;
  u32 index_offset;
  u32  triangle_count;
  u32  vertex_count;
  float4 sphere;
  float4 cone_apex;
  float4 cone_axis_cutoff;
  u32 cone_pack;
};

struct Vertex {
  float3 position;
  float3 normal;
};

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 1)
  (binding 0)
  (type Meshlet)
  (name meshlet_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 0)
  (type u32)
  (name index_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 1)
  (type float3)
  (name position_buffer)
)

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 2)
  (binding 2)
  (type float3)
  (name normal_buffer)
)

Vertex fetch(u32 index) {
  Vertex o;  
  o.position = buffer_load(position_buffer, index);
  o.normal = buffer_load(normal_buffer, index);
  return o;
}

u32 fetch_index(u32 index) {
  u32 raw = buffer_load(index_buffer, (index_offset + index) / 4);
  u32 sub_index = index & 0x3;
  return (raw >> (sub_index * 8)) & 0xffu;
}

//shared u32    gs_indices[64];
//shared float3 gs_vertices[64];
//shared Meshlet meshlet;

@(GROUP_SIZE 256 1 1)
@(ENTRY)
  
  u32 meshlet_index = GROUPT_INDEX.x;
  if (meshlet_index > num_meshlets)
    return;

  //if (LOCAL_THREAD_INDEX.x == 0)
  Meshlet  meshlet = buffer_load(meshlet_buffer, meshlet_offset + meshlet_index);
  //barrier();
  {
    float4 cp = mul4(viewproj, float4(meshlet.cone_apex.xyz, 1.0));
    float4 ca = mul4(viewproj, float4(meshlet.cone_apex.xyz + meshlet.cone_axis_cutoff.xyz, 1.0));
    if (ca.z > cp.z)
      return;
  }
  /*if (LOCAL_THREAD_INDEX.x < meshlet.vertex_count)
    gs_vertices[LOCAL_THREAD_INDEX.x] = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + LOCAL_THREAD_INDEX.x);

  barrier();*/

  u32 triangle_index = LOCAL_THREAD_INDEX.x;

  if (triangle_index > meshlet.triangle_count)
    return;

  u32 i0 = fetch_index(meshlet.index_offset + triangle_index * 3 + 0);
  u32 i1 = fetch_index(meshlet.index_offset + triangle_index * 3 + 1);
  u32 i2 = fetch_index(meshlet.index_offset + triangle_index * 3 + 2);
   
  // Vertex v0 = gs_vertices[i0];//fetch(vertex_offset + meshlet.vertex_offset + i0);
  // Vertex v0 = fetch(vertex_offset + meshlet.vertex_offset + i0);
  // Vertex v1 = gs_vertices[i1];//fetch(vertex_offset + meshlet.vertex_offset + i1);
  // Vertex v1 = fetch(vertex_offset + meshlet.vertex_offset + i1);
  // Vertex v2 = gs_vertices[i2];//fetch(vertex_offset + meshlet.vertex_offset + i2);
  // Vertex v2 = fetch(vertex_offset + meshlet.vertex_offset + i2);
  
  //float3 p0 = gs_vertices[i0];
  //float3 p1 = gs_vertices[i1];
  //float3 p2 = gs_vertices[i2];

  float3 p0 = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + i0);
  float3 p1 = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + i1);
  float3 p2 = buffer_load(position_buffer, vertex_offset + meshlet.vertex_offset + i2);

  float4 pp0 = mul4(viewproj, float4(p0, 1.0));
  float4 pp1 = mul4(viewproj, float4(p1, 1.0));
  float4 pp2 = mul4(viewproj, float4(p2, 1.0));
  pp0.xyz /= pp0.w;
  pp1.xyz /= pp1.w;
  pp2.xyz /= pp2.w;
  {
    float2 e0 = pp0.xy - pp2.xy; 
    float2 e1 = pp1.xy - pp0.xy; 
    float2 e2 = pp2.xy - pp1.xy; 
    float2 n0 = float2(e0.y, -e0.x);
    if (dot(e1, n0) > 0.0)
      return;
  }
  float area = 1.0;
  float b0 = 1.0 / 3.0;
  float b1 = 1.0 / 3.0;
  float b2 = 1.0 / 3.0;
  b0 /= area;
  b1 /= area;
  b2 /= area;
  float z = 1.0 / (b0 / pp0.w + b1 / pp1.w + b2 / pp2.w);

  float3 pp = pp0.xyz * b0 + pp1.xyz * b1 + pp2.xyz * b2;
  
  int2 dim = imageSize(out_image);
  if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
    return;
  i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
  i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
  if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
    float3 n0 = buffer_load(normal_buffer, vertex_offset + meshlet.vertex_offset + i0);
    float3 n1 = buffer_load(normal_buffer, vertex_offset + meshlet.vertex_offset + i1);
    float3 n2 = buffer_load(normal_buffer, vertex_offset + meshlet.vertex_offset + i2);
    float3 n = normalize(z * (n0 * b0 + n1 * b1 + n1 * b2));
    float4 color = float4(
        float(((31 * meshlet.vertex_offset) >> 0) % 255) / 255.0,
        float(((31 * meshlet.vertex_offset) >> 8) % 255) / 255.0,
        float(((31 * meshlet.vertex_offset) >> 16) % 255) / 255.0,
        1.0
        );
    //color = float4((0.5 * n + float3_splat(0.5)), 1.0);
    color = float4_splat(1.0) * (0.5 + 0.5 * dot(n, normalize(float3(1.0, 1.0, 1.0))));
    u32 depth = u32(1.0 / pp.z);
    if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
      image_store(out_image, int2(x, y), color);
    }
  }
@(END)
)"),
                                 NULL, 0);
    if (clear_cs.is_null())
      clear_cs = pc->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
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
@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  image_store(out_image, int2(GLOBAL_THREAD_INDEX.xy), float4(0.0, 0.0, 0.0, 1.0));
  image_store(out_depth, int2(GLOBAL_THREAD_INDEX.xy), uint4(1 << 31, 0, 0, 0));
@(END)
)"),
                                       NULL, 0);
    rd::Image_Info info;
    MEMZERO(info);
    if (output_image.is_null() == false)
      info = pc->get_image_info(output_image);
    if (output_image.is_null() ||
        g_config.get_u32("g_buffer_width") != info.width ||
        g_config.get_u32("g_buffer_height") != info.height) {
      if (output_image.is_null() == false) pc->release_resource(output_image);
      if (output_depth.is_null() == false) pc->release_resource(output_depth);
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::RGBA32_FLOAT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_image = pc->create_image(info);
      }
      {
        rd::Image_Create_Info info;
        MEMZERO(info);
        info.format     = rd::Format::R32_UINT;
        info.width      = g_config.get_u32("g_buffer_width");
        info.height     = g_config.get_u32("g_buffer_height");
        info.depth      = 1;
        info.layers     = 1;
        info.levels     = 1;
        info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        output_depth = pc->create_image(info);
      }
      pc->assign_name(output_image, stref_s("meshlet_render/img0"));
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    if (g_config.get_bool("enable_meshlets_render_pass") == false) return;
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->CS_set_shader(clear_cs);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_depth,
                       (u32)rd::Access_Bits::SHADER_WRITE |
                           (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 0, 0, output_image, 0, 1, 0, 1);
    ctx->bind_rw_image(0, 1, 0, output_depth, 0, 1, 0, 1);
    ctx->dispatch((g_config.get_u32("g_buffer_width") + 15) / 16,
                  (g_config.get_u32("g_buffer_height") + 15) / 16, 1);
    ctx->CS_set_shader(cs);
    float4x4 vp = g_camera.viewproj();
    ctx->push_constants(&vp, 0, sizeof(vp));
    g_scene.gfx_dispatch_meshlets(ctx, sizeof(vp));
  }
  string_ref get_name() override { return stref_s("meshlet_render_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(cs);
    rm->release_resource(output_image);
  }
};

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
    ctx->image_barrier(input_image, (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->image_barrier(output_image, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
    ctx->bind_rw_image(0, 1, 0, input_image, 0, 1, 0, 1);
    ctx->bind_rw_image(0, 2, 0, output_image, 0, 1, 0, 1);
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

class GUI_Pass : public rd::IPass, public rd::IEvent_Consumer {
  Resource_ID vs;
  Resource_ID ps;
  u32         width, height;
  Resource_ID sampler;
  Resource_ID vertex_buffer;
  Resource_ID index_buffer;

  Resource_ID font_texture;
  Resource_ID staging_buffer;

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
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
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
    ImGuiIO &   io               = ImGui::GetIO();
    static bool show_demo_window = true;
    ImGui::ShowDemoWindow(&show_demo_window);
    {
      ImGui::Begin("Config");
      g_config.on_imgui();
      ImGui::LabelText("hw rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("opaque_pass")));
      ImGui::LabelText("sw rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("compute_render_pass")));
      ImGui::LabelText("meshlet rasterizer", "%f ms",
                       pc->get_pass_duration(stref_s("meshlet_render_pass")));
      ImGui::End();
    }
    {
      ImGui::Begin("sw rasterization");
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("compute_render/img0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }
      ImGui::End();
      ImGui::Begin("hw rasterization");
      auto  wpos        = ImGui::GetCursorScreenPos();
      auto  wsize       = ImGui::GetWindowSize();
      float height_diff = 24;
      if (wsize.y < height_diff + 2) {
        wsize.y = 2;
      } else {
        wsize.y = wsize.y - height_diff;
      }
      // g_config.get_u32("g_buffer_width")  = wsize.x;
      // g_config.get_u32("g_buffer_height") = wsize.y;
      if (ImGui::IsWindowHovered()) {
        auto scroll_y = ImGui::GetIO().MouseWheel;
        if (scroll_y) {
          g_camera.distance += g_camera.distance * 2.e-1 * scroll_y;
          g_camera.distance = clamp(g_camera.distance, 1.0e-3f, 1000.0f);
        }
        f32 camera_speed = 2.0f;
        if (ImGui::GetIO().KeysDown[SDL_SCANCODE_LSHIFT]) {
          camera_speed = 20.0f;
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
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("opaque_pass/rt0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
      ImGui::Begin("meshlet rasterization");
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("meshlet_render/img0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }

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

      {
        ImDrawVert *vtx_dst = (ImDrawVert *)ctx->map_buffer(vertex_buffer);

        ito(draw_data->CmdListsCount) {

          const ImDrawList *cmd_list = draw_data->CmdLists[i];
          memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
                 cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));

          vtx_dst += cmd_list->VtxBuffer.Size;
        }
        ctx->unmap_buffer(vertex_buffer);
      }
      {
        ImDrawIdx *idx_dst = (ImDrawIdx *)ctx->map_buffer(index_buffer);
        ito(draw_data->CmdListsCount) {

          const ImDrawList *cmd_list = draw_data->CmdLists[i];

          memcpy(idx_dst, cmd_list->IdxBuffer.Data,
                 cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
          idx_dst += cmd_list->IdxBuffer.Size;
        }
        ctx->unmap_buffer(index_buffer);
      }
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
  pmng->add_pass(rd::Pass_t::RENDER, new Opaque_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Compute_Render_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Meshlet_Compute_Render_Pass);
  pmng->add_pass(rd::Pass_t::COMPUTE, new Postprocess_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, gui);
  pmng->loop();
  return 0;
}
