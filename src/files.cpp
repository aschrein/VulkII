#include "scene.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#include <functional>

#include <meshoptimizer.h>

void optimize_mesh(Raw_Mesh_Opaque &opaque_mesh) {
  // ASSERT_ALWAYS(opaque_mesh.index_type == rd::Index_t::UINT32);
  u32        index_count = opaque_mesh.num_indices;
  Array<u32> indices;
  indices.init();
  defer(indices.release());
  indices.resize(opaque_mesh.num_indices);
  Array<u32> remap;
  remap.init();
  remap.resize(opaque_mesh.num_indices);
  defer(remap.release());
  Array<u8> vertex_blob;
  vertex_blob.init();
  vertex_blob.resize(opaque_mesh.num_indices * sizeof(Vertex_Full));

  defer(vertex_blob.release());

  ito(opaque_mesh.num_indices / 3) {
    Triangle_Full tri = opaque_mesh.fetch_triangle(i);
    memcpy(&vertex_blob[(i * 3 + 0) * sizeof(Vertex_Full)], &tri.v0,
           sizeof(Vertex_Full));
    memcpy(&vertex_blob[(i * 3 + 1) * sizeof(Vertex_Full)], &tri.v1,
           sizeof(Vertex_Full));
    memcpy(&vertex_blob[(i * 3 + 2) * sizeof(Vertex_Full)], &tri.v2,
           sizeof(Vertex_Full));
  }
  size_t vertex_count =
      meshopt_generateVertexRemap(&remap[0], NULL, index_count, &vertex_blob[0],
                                  opaque_mesh.num_indices, sizeof(Vertex_Full));
  meshopt_remapIndexBuffer(&indices[0], NULL, opaque_mesh.num_indices,
                           &remap[0]);
  Array<Vertex_Full> vertices;
  vertices.init();
  vertices.resize(vertex_count);
  defer(vertices.release());
  meshopt_remapVertexBuffer(&vertices[0], &vertex_blob[0], index_count,
                            sizeof(Vertex_Full), &remap[0]);
  meshopt_optimizeVertexCache(&indices[0], &indices[0], index_count,
                              vertex_count);
  meshopt_optimizeVertexFetch(&vertices[0], &indices[0], index_count,
                              &vertices[0], vertex_count, sizeof(Vertex_Full));

  opaque_mesh.attribute_data.release();
  opaque_mesh.index_type = rd::Index_t::UINT32;
  opaque_mesh.index_data.release();
  opaque_mesh.index_data.resize(indices.size * 4);
  ito(indices.size) {
    u32 index = indices[i];
    memcpy(&opaque_mesh.index_data[i * 4], &index, 4);
  }
  InlineArray<size_t, 16> attribute_offsets;
  InlineArray<size_t, 16> attribute_sizes;
  InlineArray<size_t, 16> attribute_cursors;
  attribute_offsets.init();
  attribute_sizes.init();
  attribute_cursors.init();

  ito(opaque_mesh.attributes.size) {
    attribute_sizes[i] = vertices.size * opaque_mesh.attributes[i].size;
  }
  u32 total_mem = 0;
  ito(opaque_mesh.attributes.size) {
    jto(i) attribute_offsets[i] += attribute_sizes[j];
    total_mem = attribute_offsets[i] + attribute_sizes[i];
  }
  opaque_mesh.attribute_data.resize(total_mem);
  ito(vertices.size) {
    Vertex_Full v = vertices[i];
    jto(opaque_mesh.attributes.size) {
      memcpy(opaque_mesh.attribute_data.ptr + attribute_offsets[j] +
                 attribute_cursors[j],
             v.get_attribute(opaque_mesh.attributes[j].type),
             opaque_mesh.attributes[j].size);
      attribute_cursors[j] += opaque_mesh.attributes[j].size;
    }
  }
  ito(opaque_mesh.attributes.size) {
    opaque_mesh.attributes[i].offset = attribute_offsets[i];
  }
  opaque_mesh.num_indices  = indices.size;
  opaque_mesh.num_vertices = vertices.size;
}

Raw_Meshlets_Opaque build_meshlets(Raw_Mesh_Opaque &opaque_mesh) {
  ASSERT_ALWAYS(opaque_mesh.index_type == rd::Index_t::UINT32);
  const size_t max_vertices  = 64;
  const size_t max_triangles = 124;

  Array<meshopt_Meshlet> meshlets;
  meshlets.init();
  defer(meshlets.release());
  meshlets.resize(meshopt_buildMeshletsBound(opaque_mesh.num_indices,
                                             max_vertices, max_triangles));

  meshlets.resize(meshopt_buildMeshlets(
      &meshlets[0], (u32 *)&opaque_mesh.index_data[0], opaque_mesh.num_indices,
      opaque_mesh.num_vertices, max_vertices, max_triangles));

  InlineArray<Array<u8>, 0x10> tmp_attributes;
  tmp_attributes.init();
  defer({ ito(opaque_mesh.attributes.size) tmp_attributes[i].release(); });
  ito(opaque_mesh.attributes.size) {
    tmp_attributes[i].reserve((opaque_mesh.get_attribute_size(i) * 3) >> 1);
  }
  auto write_attribute_data = [&](u8 const *src, size_t size,
                                  u32 attribute_id) {
    tmp_attributes[attribute_id].reserve(tmp_attributes[attribute_id].size +
                                         size);
    memcpy(tmp_attributes[attribute_id].ptr + tmp_attributes[attribute_id].size,
           src, size);
    tmp_attributes[attribute_id].size += size;
  };
  u32                 vertex_offset = 0;
  u32                 index_offset  = 0;
  Raw_Meshlets_Opaque result;
  result.init();
  auto write_vertex = [&](u32 vertex_index) {
    ito(opaque_mesh.attributes.size) {
      write_attribute_data(opaque_mesh.get_attribute_data(i, vertex_index),
                           opaque_mesh.attributes[i].size, i);
    }
    vertex_offset += 1;
  };
  auto write_index = [&](u8 index) {
    result.index_data.reserve(result.index_data.size + 1);
    memcpy(result.index_data.ptr + index_offset, &index, 1);
    result.index_data.size += 1;
    index_offset += 1;
  };

  result.index_data.reserve(meshlets.size * 128 * 4);
  result.meshlets.reserve(meshlets.size);
  ito(meshlets.size) {
    meshopt_Meshlet meshlet = meshlets[i];

    u32 meshlet_vertex_offset = vertex_offset;
    u32 meshlet_index_offset  = index_offset;

    for (unsigned int i = 0; i < meshlet.vertex_count; ++i)
      write_vertex(meshlet.vertices[i]);

    ito(meshlet.triangle_count) {
      write_index(meshlet.indices[i][0]);
      write_index(meshlet.indices[i][1]);
      write_index(meshlet.indices[i][2]);
    }
    Attribute position_attribute =
        opaque_mesh.get_attribute(rd::Attriute_t::POSITION);

    meshopt_Bounds bounds = meshopt_computeMeshletBounds(
        &meshlet,
        (float *)&opaque_mesh.attribute_data[position_attribute.offset],
        opaque_mesh.num_vertices, position_attribute.stride);

    Meshlet m        = {};
    m.vertex_offset  = meshlet_vertex_offset;
    m.index_offset   = meshlet_index_offset;
    m.triangle_count = meshlet.triangle_count;
    m.vertex_count   = meshlet.vertex_count;

    m.sphere = float4(bounds.center[0], bounds.center[1], bounds.center[2],
                      bounds.radius);

    m.cone_apex.x = bounds.cone_apex[0];
    m.cone_apex.y = bounds.cone_apex[1];
    m.cone_apex.z = bounds.cone_apex[2];

    m.cone_axis_cutoff.x = bounds.cone_axis[0];
    m.cone_axis_cutoff.y = bounds.cone_axis[1];
    m.cone_axis_cutoff.z = bounds.cone_axis[2];
    m.cone_axis_cutoff.w = bounds.cone_cutoff;

    m.cone_axis_s8[0] = bounds.cone_axis_s8[0];
    m.cone_axis_s8[1] = bounds.cone_axis_s8[1];
    m.cone_axis_s8[2] = bounds.cone_axis_s8[2];
    m.cone_cutoff_s8  = bounds.cone_cutoff_s8;

    result.meshlets.push(m);
  }

  u32 total_mem = 0;
  ito(opaque_mesh.attributes.size) { total_mem += tmp_attributes[i].size; }
  result.attribute_data.reserve(total_mem);
  result.attributes.resize(opaque_mesh.attributes.size);
  ito(opaque_mesh.attributes.size) {
    memcpy(result.attribute_data.ptr + result.attribute_data.size,
           tmp_attributes[i].ptr, tmp_attributes[i].size);
    result.attributes[i]        = opaque_mesh.attributes[i];
    result.attributes[i].offset = result.attribute_data.size;
    result.attributes[i].stride = opaque_mesh.attributes[i].size;
    result.attribute_data.size += tmp_attributes[i].size;
  }
  while (result.meshlets.size % 32 != 0) result.meshlets.push(Meshlet{});
  result.num_vertices = vertex_offset;
  result.num_indices  = index_offset;

  return result;
}

Image2D_Raw load_image(string_ref filename, rd::Format format) {
  TMP_STORAGE_SCOPE;
  if (stref_find(filename, stref_s(".hdr")) != -1) {
    int            width, height, channels;
    unsigned char *result;
    FILE *         f = stbi__fopen(stref_to_tmp_cstr(filename), "rb");
    ASSERT_PANIC(f);
    stbi__context s;
    stbi__start_file(&s, f);
    stbi__result_info ri;
    memset(&ri, 0, sizeof(ri));
    ri.bits_per_channel = 8;
    ri.channel_order    = STBI_ORDER_RGB;
    ri.num_channels     = 0;
    float *hdr = stbi__hdr_load(&s, &width, &height, &channels, STBI_rgb, &ri);

    fclose(f);
    ASSERT_PANIC(hdr);
    Image2D_Raw out;
    out.init(width, height, rd::Format::RGB32_FLOAT, (u8 *)hdr);
    stbi_image_free(hdr);
    return out;
  } else {
    int  width, height, channels;
    auto image = stbi_load(stref_to_tmp_cstr(filename), &width, &height,
                           &channels, STBI_rgb_alpha);
    ASSERT_PANIC(image);
    Image2D_Raw out;
    out.init(width, height, format, image);
    stbi_image_free(image);
    return out;
  }
}

Node *load_gltf_pbr(IFactory *factory, string_ref filename) {
  string_ref dir_path = get_dir(filename);
  TMP_STORAGE_SCOPE;
  cgltf_options options;
  MEMZERO(options);
  cgltf_data *data = NULL;
  ASSERT_ALWAYS(cgltf_parse_file(&options, stref_to_tmp_cstr(filename),
                                 &data) == cgltf_result_success);
  defer(cgltf_free(data));
  ASSERT_ALWAYS(
      cgltf_load_buffers(&options, data, stref_to_tmp_cstr(filename)) ==
      cgltf_result_success);
  Hash_Table<string_ref, i32>   loaded_textures;
  Hash_Table<cgltf_mesh *, Mesh*> mesh_table;
  loaded_textures.init();
  mesh_table.init();
  defer(loaded_textures.release());
  defer({ mesh_table.release(); });
  auto load_texture = [&](char const *uri, rd::Format format) {
    if (uri == NULL) return -1;
    if (loaded_textures.contains(stref_s(uri))) {
      return loaded_textures.get(stref_s(uri));
    }
    char full_path[0x100];
    snprintf(full_path, sizeof(full_path), "%.*s/%s", STRF(dir_path), uri);
    u32 img_id = factory->add_image(load_image(stref_s(full_path), format));
    loaded_textures.insert(stref_s(uri), (i32)img_id);
    return (i32)img_id;
  };
  for (u32 mesh_index = 0; mesh_index < data->meshes_count; mesh_index++) {

    cgltf_mesh *mesh = &data->meshes[mesh_index];
    // MeshNode *  mnode = factory->add_mesh_node(stref_s(mesh->name));
    Mesh *mymesh = factory->add_mesh(stref_s(mesh->name));
    mesh_table.insert(mesh, mymesh);
    for (u32 primitive_index = 0; primitive_index < mesh->primitives_count;
         primitive_index++) {
      cgltf_primitive *primitive = &mesh->primitives[primitive_index];
      ASSERT_ALWAYS(primitive->has_draco_mesh_compression == false);
      ASSERT_ALWAYS(primitive->extensions_count == 0);
      ASSERT_ALWAYS(primitive->targets_count == 0);
      // Load material
      PBR_Material pbrmat;
      pbrmat.init();
      {
        cgltf_material *material = primitive->material;
        ASSERT_ALWAYS(material->extensions_count == 0);

        if (material->has_pbr_metallic_roughness) {
          if (material->pbr_metallic_roughness.base_color_texture.texture !=
              NULL)
            pbrmat.albedo_id =
                load_texture(material->pbr_metallic_roughness.base_color_texture
                                 .texture->image->uri,
                             rd::Format::RGBA8_SRGBA);
          if (material->pbr_metallic_roughness.metallic_roughness_texture
                  .texture != NULL)
            pbrmat.arm_id = load_texture(
                material->pbr_metallic_roughness.metallic_roughness_texture
                    .texture->image->uri,
                rd::Format::RGBA8_UNORM);
          pbrmat.albedo_factor =
              float4(material->pbr_metallic_roughness.base_color_factor[0], //
                     material->pbr_metallic_roughness.base_color_factor[1], //
                     material->pbr_metallic_roughness.base_color_factor[2], //
                     material->pbr_metallic_roughness.base_color_factor[3]  //
              );
          pbrmat.metal_factor =
              material->pbr_metallic_roughness.metallic_factor;
          pbrmat.roughness_factor =
              material->pbr_metallic_roughness.roughness_factor;
        } else {
          if (material->pbr_specular_glossiness.diffuse_texture.texture != NULL)
            pbrmat.albedo_id =
                load_texture(material->pbr_specular_glossiness.diffuse_texture
                                 .texture->image->uri,
                             rd::Format::RGBA8_SRGBA);
          if (material->pbr_specular_glossiness.specular_glossiness_texture
                  .texture != NULL)
            pbrmat.arm_id = load_texture(
                material->pbr_specular_glossiness.specular_glossiness_texture
                    .texture->image->uri,
                rd::Format::RGBA8_UNORM);
          pbrmat.metal_factor =
              material->pbr_specular_glossiness.specular_factor[0];
          pbrmat.roughness_factor =
              material->pbr_specular_glossiness.glossiness_factor;
        }
        if (material->normal_texture.texture != NULL)
          pbrmat.normal_id =
              load_texture(material->normal_texture.texture->image->uri,
                           rd::Format::RGBA8_UNORM);
      }
      Raw_Mesh_Opaque opaque_mesh;
      opaque_mesh.init();
      // Read indices
      u32 index_stride = 0;
      if (primitive->indices->component_type == cgltf_component_type_r_16u) {
        opaque_mesh.index_type = rd::Index_t::UINT16;
        index_stride           = 2;
      } else if (primitive->indices->component_type ==
                 cgltf_component_type_r_32u) {
        opaque_mesh.index_type = rd::Index_t::UINT32;
        index_stride           = 4;
      } else {
        UNIMPLEMENTED;
      }
      opaque_mesh.index_data.reserve(primitive->indices->count * index_stride);
      opaque_mesh.num_indices = primitive->indices->count;
      auto write_index_data   = [&](u8 *src, size_t size) {
        ito(size) opaque_mesh.index_data.push(src[i]);
      };
      if (primitive->indices->component_type == cgltf_component_type_r_16u) {
        ito(primitive->indices->count) {
          cgltf_size index = cgltf_accessor_read_index(primitive->indices, i);
          u16        cv    = (u16)index;
          write_index_data((u8 *)&cv, 2);
        }
      } else if (primitive->indices->component_type ==
                 cgltf_component_type_r_32u) {
        ito(primitive->indices->count) {
          cgltf_size index = cgltf_accessor_read_index(primitive->indices, i);
          u32        cv    = (u32)index;
          write_index_data((u8 *)&cv, 4);
        }
      } else {
        UNIMPLEMENTED;
      }
      // Read attributes
      opaque_mesh.max = vec3(-1.0e10f);
      opaque_mesh.min = vec3(1.0e10f);
      opaque_mesh.attribute_data.reserve(primitive->indices->count *
                                         sizeof(Vertex_Full));
      auto write_attribute_data = [&](u8 *src, size_t size) {
        ito(size) opaque_mesh.attribute_data.push(src[i]);
      };
      auto align_attribute_data = [&]() {
        while ((opaque_mesh.attribute_data.size & 0xff) != 0)
          opaque_mesh.attribute_data.push(0);
      };

      for (u32 attribute_index = 0;
           attribute_index < primitive->attributes_count; attribute_index++) {
        cgltf_attribute *attribute = &primitive->attributes[attribute_index];
        align_attribute_data();
        if (opaque_mesh.num_vertices == 0)
          opaque_mesh.num_vertices = attribute->data->count;
        else {
          ASSERT_ALWAYS(attribute->data->count == opaque_mesh.num_vertices);
        }
        ASSERT_ALWAYS(attribute->data->is_sparse == false);
        ASSERT_ALWAYS(attribute->data->extensions_count == 0);
        ASSERT_ALWAYS(attribute->data->normalized == false);
        switch (attribute->type) {
        case cgltf_attribute_type_position: {
          ASSERT_ALWAYS(attribute->index == 0);
          ASSERT_ALWAYS(attribute->data->stride == 12);
          ASSERT_ALWAYS(attribute->data->component_type ==
                        cgltf_component_type_r_32f);
          Attribute a;
          MEMZERO(a);
          a.format = rd::Format::RGB32_FLOAT;
          a.offset = opaque_mesh.attribute_data.size;
          a.stride = 12;
          a.size   = 12;
          a.type   = rd::Attriute_t::POSITION;
          opaque_mesh.attributes.push(a);

          ito(attribute->data->count) {
            float pos[3];
            cgltf_accessor_read_float(attribute->data, i, pos, 3);
            pos[0] *= -0.04f;
            pos[1] *= -0.04f;
            pos[2] *= 0.04f;
            opaque_mesh.max.x = MAX(opaque_mesh.max.x, pos[0]);
            opaque_mesh.max.y = MAX(opaque_mesh.max.y, pos[1]);
            opaque_mesh.max.z = MAX(opaque_mesh.max.z, pos[2]);
            opaque_mesh.min.x = MIN(opaque_mesh.min.x, pos[0]);
            opaque_mesh.min.y = MIN(opaque_mesh.min.y, pos[1]);
            opaque_mesh.min.z = MIN(opaque_mesh.min.z, pos[2]);
            write_attribute_data((u8 *)pos, 12);
          }
          break;
        }
        case cgltf_attribute_type_normal: {
          ASSERT_ALWAYS(attribute->index == 0);
          ASSERT_ALWAYS(attribute->data->stride == 12);
          ASSERT_ALWAYS(attribute->data->component_type ==
                        cgltf_component_type_r_32f);
          Attribute a;
          MEMZERO(a);
          a.format = rd::Format::RGB32_FLOAT;
          a.offset = opaque_mesh.attribute_data.size;
          a.stride = 12;
          a.size   = 12;
          a.type   = rd::Attriute_t::NORMAL;
          opaque_mesh.attributes.push(a);
          ito(attribute->data->count) {
            float pos[3];
            cgltf_accessor_read_float(attribute->data, i, pos, 3);
            write_attribute_data((u8 *)pos, 12);
          }
          break;
        }
        case cgltf_attribute_type_tangent: {
          ASSERT_ALWAYS(attribute->index == 0);
          ASSERT_ALWAYS(attribute->data->stride == 16);
          ASSERT_ALWAYS(attribute->data->component_type ==
                        cgltf_component_type_r_32f);
          Attribute a;
          MEMZERO(a);
          a.format = rd::Format::RGBA32_FLOAT;
          a.offset = opaque_mesh.attribute_data.size;
          a.stride = 16;
          a.size   = 16;
          a.type   = rd::Attriute_t::TANGENT;
          opaque_mesh.attributes.push(a);
          ito(attribute->data->count) {
            float pos[4];
            cgltf_accessor_read_float(attribute->data, i, pos, 4);
            write_attribute_data((u8 *)pos, 16);
          }
          break;
        }
        case cgltf_attribute_type_texcoord: {
          if (attribute->index >= 4) continue;
          ASSERT_ALWAYS(attribute->index < 4);
          ASSERT_ALWAYS(attribute->data->stride == 8);
          ASSERT_ALWAYS(attribute->data->component_type ==
                        cgltf_component_type_r_32f);
          Attribute a;
          MEMZERO(a);
          a.format = rd::Format::RG32_FLOAT;
          a.offset = opaque_mesh.attribute_data.size;
          a.stride = 8;
          a.size   = 8;
          if (attribute->index == 0)
            a.type = rd::Attriute_t::TEXCOORD0;
          else if (attribute->index == 1)
            a.type = rd::Attriute_t::TEXCOORD1;
          else if (attribute->index == 2)
            a.type = rd::Attriute_t::TEXCOORD2;
          else if (attribute->index == 3)
            a.type = rd::Attriute_t::TEXCOORD3;
          else
            UNIMPLEMENTED;
          opaque_mesh.attributes.push(a);
          ito(attribute->data->count) {
            float pos[2];
            cgltf_accessor_read_float(attribute->data, i, pos, 2);
            write_attribute_data((u8 *)pos, 8);
          }

          break;
        }
        case cgltf_attribute_type_weights:
        case cgltf_attribute_type_joints: {
          // Skip
          break;
        }
        default: UNIMPLEMENTED;
        }
      }
      opaque_mesh.sort_attributes();
      optimize_mesh(opaque_mesh);
      Raw_Meshlets_Opaque meshlets = build_meshlets(opaque_mesh);
      mymesh->add_primitive(opaque_mesh, meshlets, pbrmat);
    }
  }
  Hash_Table<cgltf_node *, Node *> node_indices;
  node_indices.init();
  defer(node_indices.release());
  std::function<Node *(cgltf_node *)> load_node = [&](cgltf_node *node) {
    if (node_indices.contains(node)) {
      return node_indices.get(node);
    }
    Node *tnode = NULL;
    if (node->mesh != NULL) {
      MeshNode *mnode = factory->add_mesh_node(stref_s(node->name));
      ASSERT_ALWAYS(mesh_table.contains(node->mesh));
      Mesh *mesh= mesh_table.get(node->mesh);
      mnode->set_mesh(mesh);
      tnode = mnode;
    } else {
      tnode = factory->add_node(stref_s(node->name));  
    }
    if (node->has_translation) {
      tnode->offset = float3(node->translation[0], node->translation[1],
                             node->translation[2]);
    }
    if (node->has_scale) {
      tnode->scale = float3(node->scale[0], node->scale[1], node->scale[2]);
    }
    if (node->has_rotation) {
      tnode->rotation = quat(node->rotation[0], node->rotation[1],
                             node->rotation[2], node->rotation[3]);
    }
    for (u32 child_index = 0; child_index < node->children_count;
         child_index++) {
      cgltf_node *child_node = node->children[child_index];
      tnode->add_child(load_node(child_node));
    }
    return tnode;
  };
  Node *root = load_node(&data->nodes[0]);

  vec3 max = vec3(-1.0e10f);
  vec3 min = vec3(1.0e10f);

  vec3 max_dims = max - min;

  // Size normalization hack
  float vk      = 1.0f;
  float max_dim = MAX3(max_dims.x, max_dims.y, max_dims.z);
  vk            = 50.0f / max_dim;
  vec3 avg      = (max + min) / 2.0f;

  //

  root->offset = -avg * vk;
  return root;
}