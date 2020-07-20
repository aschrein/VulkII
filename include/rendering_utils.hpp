#ifndef RENDERING_UTILS_HPP
#define RENDERING_UTILS_HPP

#include "rendering.hpp"
#include "scene.hpp"
#include "script.hpp"
#include "utils.hpp"
#include <imgui.h>

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

#include <imgui.h>
#include <imgui/examples/imgui_impl_sdl.h>

struct Config_Item {
  enum Type { U32, F32, BOOL };
  Type type;
  union {
    u32  v_u32;
    f32  v_f32;
    bool v_bool;
  };
  union {
    u32 v_u32_min;
    u32 v_f32_min;
  };
  union {
    u32 v_u32_max;
    u32 v_f32_max;
  };
};

struct Tmp_List_Allocator {
  List *alloc() {
    List *out = (List *)tl_alloc_tmp(sizeof(List));
    memset(out, 0, sizeof(List));
    return out;
  }
};

struct Config {
  using string_t = inline_string<32>;
  Hash_Table<string_t, Config_Item> items;

  void init(string_ref init_script) {
    items.init();
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(init_script, Tmp_List_Allocator());
    traverse(cur);
  }

  void release() { items.release(); }

  void traverse(List *l) {
    struct Params {
      i32  imin = -1;
      f32  fmin = -1;
      i32  imax = -1;
      f32  fmax = -1;
      void traverse(List *l) {
        if (l == NULL) return;
        if (l->child) {
          traverse(l->child);
          traverse(l->next);
        } else {
          if (l->cmp_symbol("min")) {
            if (parse_decimal_int(l->get(1)->symbol.ptr, l->get(1)->symbol.len,
                                  &imin) == false)
              parse_float(l->get(1)->symbol.ptr, l->get(1)->symbol.len, &fmin);
          } else if (l->cmp_symbol("max")) {
            if (parse_decimal_int(l->get(1)->symbol.ptr, l->get(1)->symbol.len,
                                  &imax) == false)
              parse_float(l->get(1)->symbol.ptr, l->get(1)->symbol.len, &fmax);
          }
        }
      }
    };
    if (l == NULL) return;
    if (l->child) {
      traverse(l->child);
      traverse(l->next);
    } else {
      if (l->cmp_symbol("add")) {
        string_ref  type = l->get(1)->symbol;
        string_ref  name = l->get(2)->symbol;
        Config_Item item;
        MEMZERO(item);
        string_t _name;
        _name.init(name);
        if (type == stref_s("u32")) {
          Params params;
          params.traverse(l->get(4));
          item.type  = Config_Item::U32;
          item.v_u32 = l->get(3)->parse_int();
          if (params.imin != params.imax) {
            item.v_u32_min = params.imin;
            item.v_u32_max = params.imax;
          }
          items.insert(_name, item);
        } else if (type == stref_s("f32")) {
          Params params;
          params.traverse(l->get(4));
          if (params.fmin != params.fmax) {
            item.v_f32_min = params.fmin;
            item.v_f32_max = params.fmax;
          }
          item.type  = Config_Item::F32;
          item.v_f32 = l->get(3)->parse_float();
          items.insert(_name, item);
        } else if (type == stref_s("bool")) {
          item.type   = Config_Item::BOOL;
          item.v_bool = l->get(3)->parse_int() > 0;
          items.insert(_name, item);
        } else {
          TRAP;
        }
      }
    }
  }

  void on_imgui() {
    items.iter_pairs([&](string_t const &name, Config_Item &item) {
      char buf[0x100];
      snprintf(buf, sizeof(buf), "%.*s", STRF(name.ref()));
      if (item.type == Config_Item::U32) {
        if (item.v_u32_min != item.v_u32_max) {
          ImGui::SliderInt(buf, (int *)&item.v_u32, item.v_u32_min,
                           item.v_u32_max);
        } else {
          ImGui::InputInt(buf, (int *)&item.v_u32);
        }
      } else if (item.type == Config_Item::F32) {
        if (item.v_f32_min != item.v_f32_max) {
          ImGui::SliderFloat(buf, (float *)&item.v_f32, item.v_f32_min,
                             item.v_f32_max);
        } else {
          ImGui::InputFloat(buf, (float *)&item.v_f32);
        }
      } else if (item.type == Config_Item::BOOL) {
        ImGui::Checkbox(buf, &item.v_bool);
      } else {
        TRAP;
      }
    });
  }

  u32 &get_u32(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    ASSERT_DEBUG(items.contains(_name));
    return items.get_ref(_name).v_u32;
  }

  f32 &get_f32(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    ASSERT_DEBUG(items.contains(_name));
    return items.get_ref(_name).v_f32;
  }

  bool &get_bool(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    ASSERT_DEBUG(items.contains(_name));
    return items.get_ref(_name).v_bool;
  }

  void dump(FILE *file) {
    fprintf(file, "(config\n");
    items.iter_pairs([&](string_t const &name, Config_Item const &item) {
      if (item.type == Config_Item::U32) {
        if (item.v_u32_min != item.v_u32_max)
          fprintf(file, " (add u32 %.*s %i (min %i) (max %i))\n",
                  STRF(name.ref()), item.v_u32, item.v_u32_min, item.v_u32_max);
        else
          fprintf(file, " (add u32 %.*s %i)\n", STRF(name.ref()), item.v_u32);
      } else if (item.type == Config_Item::F32) {
        if (item.v_f32_min != item.v_f32_max)
          fprintf(file, " (add f32 %.*s %f (min %f) (max %f))\n",
                  STRF(name.ref()), item.v_f32, item.v_f32_min, item.v_f32_max);
        else
          fprintf(file, " (add f32 %.*s %f)\n", STRF(name.ref()), item.v_f32);
      } else if (item.type == Config_Item::BOOL) {
        fprintf(file, " (add bool %.*s %i)\n", STRF(name.ref()),
                item.v_bool ? 1 : 0);
      } else {
        TRAP;
      }
    });
    fprintf(file, ")\n");
  }
};

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
    if (l == NULL) return;
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
};

struct GfxMesh : public Mesh {
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

  void init(string_ref name) {
    Mesh::init(name);
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

  static GfxMesh *create(string_ref name) {
    GfxMesh *out = new GfxMesh;
    out->init(name);
    return out;
  }
  static u64 ID() {
    static char p;
    return (u64)(intptr_t)&p;
  }
  void release() {
    Mesh::release();
    delete this;
  }
  u32 get_num_indices() {
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
      ctx->dispatch((primitives[i].mesh.num_indices + 255) / 256, 1, 1);
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
  public:
  Node *             root;
  Array<Image2D_Raw> images;
  Array<GfxMesh *>   meshes;

  // GFX state cache
  Array<Resource_ID>            gfx_images;
  Hash_Table<GfxMesh *, size_t> mesh_offsets;
  bool                          gfx_buffers_initialized;
  u32                           texture_streaming_id;
  size_t                        total_memory;
  Resource_ID                   vertex_buffer;
  Resource_ID                   staging_vertex_buffer;
  Resource_ID                   staging_buffer;
  Resource_ID                   dummy_texture;
  bool                          dummy_initialized;

  friend class SceneFactory;
  class SceneFactory : public IFactory {
public:
    Scene *scene;

    SceneFactory(Scene *scene) : scene(scene) {}
    Node *    add_node(string_ref name) override { return Node::create(name); }
    MeshNode *add_mesh_node(string_ref name) override {
      MeshNode *model = MeshNode::create(name);
      return model;
    }
    Mesh *add_mesh(string_ref name) override {
      GfxMesh *mesh = GfxMesh::create(name);
      scene->meshes.push(mesh);
      return mesh;
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

    root = Node::create(stref_s("ROOT"));
  }

  void load_mesh(string_ref name, string_ref path) {
    SceneFactory sf(this);
    root->add_child(
        load_gltf_pbr(&sf, path) // stref_s("models/light/scene_low.gltf"))
            ->rename(name));
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
    root->update_transform();
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
    if (!gfx_buffers_initialized) {
      gfx_buffers_initialized = true;
      ito(meshes.size) {
        size_t mesh_mem = meshes[i]->get_needed_memory();
        mesh_mem        = rd::IResource_Manager::align_up(mesh_mem);
        mesh_offsets.insert(meshes[i], total_memory);
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
        ito(meshes.size) {
          meshes[i]->put_data(ptr + mesh_offsets.get(meshes[i]));
        }
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
    traverse(
        [&](Node *node) {
          if (isa<MeshNode>(node) && node->get_name() == stref_s("HIGH")) {
            GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
            ctx->push_constants(&node->get_transform(), 0, sizeof(float4x4));
            gfxmesh->draw(ctx, vertex_buffer, mesh_offsets.get(gfxmesh));
          }
        },
        root);
  }
  void gfx_exec_low(rd::Imm_Ctx *ctx) {
    gfx_bind(ctx);
    traverse(
        [&](Node *node) {
          if (isa<MeshNode>(node) && node->get_name() == stref_s("LOW")) {
            GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
            ctx->push_constants(&node->get_transform(), 0, sizeof(float4x4));
            ((GfxMesh *)((MeshNode *)node)->get_mesh())
                ->draw(ctx, vertex_buffer,
                       mesh_offsets.get(
                           ((GfxMesh *)((MeshNode *)node)->get_mesh())));
          }
        },
        root);
  }
  void gfx_dispatch(rd::Imm_Ctx *ctx, u32 push_offset) {
    gfx_bind(ctx);
    traverse(
        [&](Node *node) {
          if (isa<MeshNode>(node) && node->get_name() == stref_s("HIGH")) {
            GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
            ctx->push_constants(&node->get_transform(), push_offset,
                                sizeof(float4x4));
            ((GfxMesh *)((MeshNode *)node)->get_mesh())
                ->dispatch(ctx, vertex_buffer,
                           mesh_offsets.get(
                               ((GfxMesh *)((MeshNode *)node)->get_mesh())),
                           push_offset + 64);
          }
        },
        root);
  }
  void gfx_dispatch_meshlets(rd::Imm_Ctx *ctx, u32 push_offset) {
    gfx_bind(ctx);
    traverse(
        [&](Node *node) {
          if (isa<MeshNode>(node) && node->get_name() == stref_s("HIGH")) {
            GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
            ctx->push_constants(&node->get_transform(), push_offset,
                                sizeof(float4x4));
            ((GfxMesh *)((MeshNode *)node)->get_mesh())
                ->dispatch_meshlets(
                    ctx, vertex_buffer,
                    mesh_offsets.get(
                        ((GfxMesh *)((MeshNode *)node)->get_mesh())),
                    push_offset + 64);
          }
        },
        root);
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
    ito(meshes.size) meshes[i]->release();
    meshes.release();
    ito(images.size) images[i].release();
    images.release();
    root->release();
  }
};

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

class IGUI_Pass : public rd::IPass, public rd::IEvent_Consumer {
  protected:
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

  static constexpr u64 DEPTH_FLAG = 0x8000'0000'0000'0000ull;

  public:
  virtual void on_gui(rd::IResource_Manager *pc) {}

  void consume(void *_event) override {
    SDL_Event *event = (SDL_Event *)_event;
    if (imgui_initialized) {
      ImGui_ImplSDL2_ProcessEvent(event);
    }
    if (event->type == SDL_MOUSEMOTION) {
      SDL_MouseMotionEvent *m = (SDL_MouseMotionEvent *)event;
    }
  }
  IGUI_Pass() {
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
  (add_field (type u32)      (name control_flags))
)

#define CONTROL_DEPTH_ENABLE 1
#define is_control(flag) (control_flags & flag) != 0

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
  if (is_control(CONTROL_DEPTH_ENABLE)) {
    float depth = texture(sampler2D(sTexture, sSampler), UV).r;
    depth = pow(depth * 500.0, 1.0 / 2.0);
    @(EXPORT_COLOR 0
      float4_splat(depth)
    );
  } else {
    @(EXPORT_COLOR 0
      Color * texture(sampler2D(sTexture, sSampler), UV)
    );
  }
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
    
    static bool show_demo_window = true;
    ImGui::ShowDemoWindow(&show_demo_window);
    on_gui(pc);
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
      u32 control = 0;
      ctx->push_constants(&control, 16, 4);
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
        u64         tex_id = (u64)pcmd->TextureId;
        res_id.data        = tex_id & (~DEPTH_FLAG);
        if ((tex_id & DEPTH_FLAG) != 0) {
          u32 control = 1;
          ctx->push_constants(&control, 16, 4);
        }
        ctx->bind_image(0, 0, 0, res_id, 0, 1, 0, 1);
        ctx->set_scissor(clip_rect.x, clip_rect.y, clip_rect.z - clip_rect.x,
                         clip_rect.w - clip_rect.y);
        ctx->draw_indexed(pcmd->ElemCount, 1,
                          pcmd->IdxOffset + global_idx_offset, 0,
                          pcmd->VtxOffset + global_vtx_offset);
        if ((tex_id & DEPTH_FLAG) != 0) {
          u32 control = 0;
          ctx->push_constants(&control, 16, 4);
        }
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

#endif // RENDERING_UTILS_HPP
