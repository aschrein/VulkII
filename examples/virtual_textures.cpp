
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

inline float  saturate(float s) { return s > 1.0f ? 1.0f : (s < 0.0f ? 0.0f : s); }
inline float3 saturate(float3 s) {
  s.x = saturate(s.x);
  s.y = saturate(s.y);
  s.z = saturate(s.z);
  return s;
}
f32 normalize_sdf(f32 v, f32 voxel_size) {
  return saturate(v / (voxel_size * sqrtf(3.0f)) * 0.5f + 0.5f);
}
u16 pack_sdf(f32 v) { return u16(double(saturate(v)) * double((1u << 16u) - 1u) + 0.5f); }
struct SDF_float8x8x8 {
  f32  sdf[8 * 8 * 8];
  void render_viz(float3 const &min, float3 const &max) {
    float3 size = max - min;
    zto(8) {
      yto(8) {
        xto(8) {
          float3 p     = min + size * float3(f32(x) / 7.0f, f32(y) / 7.0f, f32(z) / 7.0f);
          float  dist  = sdf[x + y * 8 + z * 8 * 8];
          dist         = dist * 2.0f - 1.0f;
          float4 color = float4(1.0f, 0.0f, 0.0f, 1.0f) * saturate(dist) +
                         float4(0.0f, 0.0f, 1.0f, 1.0f) * saturate(-dist);
          gizmo_layer->draw_sphere(p, 0.1f, color.xyz);
        }
      }
    }
  }
  void render_values_viz(float3 const &min, float3 const &max) {
    float3 size       = max - min;
    float  dist_scale = size.x;
    char   buf[0x100];
    zto(8) {
      yto(8) {
        xto(8) {
          float3 p    = min + size * float3(f32(x) / 7.0f, f32(y) / 7.0f, f32(z) / 7.0f);
          float  dist = sdf[x + y * 8 + z * 8 * 8];
          dist        = dist * 2.0f - 1.0f;
          dist *= sqrtf(3.0f);
          // dist *= dist_scale;
          snprintf(buf, sizeof(buf), "%f", dist);
          gizmo_layer->draw_string(stref_s(buf), p, float3(1.0f, 1.0f, 1.0f));
        }
      }
    }
  }
  float sample_debug(float3 const &min, float3 const &max, float3 const &wp, float3 p) {
    p = p * 0.5f + float3(0.5f, 0.5f, 0.5f);
    p = saturate(p) * 7.0f;

    int x[2];
    int y[2];
    int z[2];

    float wx[2];
    float wy[2];
    float wz[2];

    x[0]  = int(p.x);
    wx[0] = p.x - x[0];
    x[1]  = int(p.x + 1.0f);
    wx[1] = x[1] - p.x;

    y[0]  = int(p.y);
    wy[0] = p.y - y[0];
    y[1]  = int(p.y + 1.0f);
    wy[1] = y[1] - p.y;

    z[0]            = int(p.z);
    wz[0]           = p.z - z[0];
    z[1]            = int(p.z + 1.0f);
    wz[1]           = z[1] - p.z;
    float  f        = 0.0f;
    f32    vals[8]  = {};
    float3 size     = (max - min);
    float3 new_size = size / 7.0f;
    ito(8) {
      i32 ix = ((i >> 0) & 1);
      i32 iy = ((i >> 1) & 1);
      i32 iz = ((i >> 2) & 1);
      {
        float3 new_center = min + float3(f32(x[ix]), f32(y[iy]), f32(z[iz])) * new_size;
        gizmo_layer->draw_line(wp, new_center, float3(1.0f, 0.0f, 0.0f));
        gizmo_layer->draw_string((wp + new_center) * 0.5f, float3(1.0f, 0.0f, 0.0f), "%f",
                                 (1.0f - wx[ix]) * (1.0f - wy[iy]) * (1.0f - wz[iz]));
      }
      if (x[ix] == 8 || y[iy] == 8 || z[iz] == 8) continue;
      vals[i] = sdf[x[ix] + y[iy] * 8 + z[iz] * 8 * 8];
    }
    ito(8) {
      i32 ix = ((i >> 0) & 1);
      i32 iy = ((i >> 1) & 1);
      i32 iz = ((i >> 2) & 1);
      f += vals[i] * (1.0f - wx[ix]) * (1.0f - wy[iy]) * (1.0f - wz[iz]);
    }
    f = (f * 2.0f - 1.0f) * sqrtf(3.0f);
    /*if (f < 0.0f) {
      ito(8) {
        i32 ix = ((i >> 0) & 1);
        i32 iy = ((i >> 1) & 1);
        i32 iz = ((i >> 2) & 1);
        fprintf(stdout, "%f * %f\n", vals[i], (1.0f - wx[ix]) * (1.0f - wy[iy]) * (1.0f - wz[iz]));
      }
      fprintf(stdout, "= %f\n", f);
    }*/

    return f;
  }
  float sample(float3 p) {
    p = p * 0.5f + float3(0.5f, 0.5f, 0.5f);
    p = saturate(p) * 7.0f;

    int x[2];
    int y[2];
    int z[2];

    float wx[2];
    float wy[2];
    float wz[2];

    x[0]  = int(p.x);
    wx[0] = p.x - x[0];
    x[1]  = int(p.x + 1.0f);
    wx[1] = x[1] - p.x;

    y[0]  = int(p.y);
    wy[0] = p.y - y[0];
    y[1]  = int(p.y + 1.0f);
    wy[1] = y[1] - p.y;

    z[0]          = int(p.z);
    wz[0]         = p.z - z[0];
    z[1]          = int(p.z + 1.0f);
    wz[1]         = z[1] - p.z;
    float f       = 0.0f;
    f32   vals[8] = {};
    ito(8) {
      i32 ix = ((i >> 0) & 1);
      i32 iy = ((i >> 1) & 1);
      i32 iz = ((i >> 2) & 1);
      if (x[ix] == 8 || y[iy] == 8 || z[iz] == 8) continue;
      vals[i] = sdf[x[ix] + y[iy] * 8 + z[iz] * 8 * 8];
    }
    ito(8) {
      i32 ix = ((i >> 0) & 1);
      i32 iy = ((i >> 1) & 1);
      i32 iz = ((i >> 2) & 1);
      f += vals[i] * (1.0f - wx[ix]) * (1.0f - wy[iy]) * (1.0f - wz[iz]);
    }
    f = (f * 2.0f - 1.0f) * sqrtf(3.0f);
    return f;
  }
}; // 32 bits / voxel

struct SDF_float8x8x8_minmax {
  AABB aabb;
  f32  min_val;
  f32  max_val;
  f32  avg_val;
  f32  voxel_size;
  f32  sdf[8 * 8 * 8];
  bool is_surface() { return min_val < 0.0 && max_val > 0.0; }
}; // 32 bits / voxel

struct SDF_half8x8x8 {
  half_float::half sdf[8 * 8 * 8];
}; // 16 bits / voxel

struct SDF_u168x8x8 {
  u16 sdf[8 * 8 * 8];
}; // 16 bits / voxel

struct SDF_byte8x8x8 {
  i8 sdf[8 * 8 * 8];
}; // 8 bits / voxel

struct SDF_diff8x8x8 {
  f32 min;
  f32 max;              // min + max = 64 bits
  u8  grade[8 * 8 * 8]; // 8 bits * 8 * 8 * 8
};

struct SDF_diff4x4x4 {
  f32 min;
  f32 max;
  u8  grade[4 * 4 * 4];
}; // 6 bits / voxel

// static Hash_Set<i32> index_set;

struct SDF_Node {
  static constexpr u32 LEAF_BIT = 1u << 31u;

  static constexpr u32 ITEMS_OFFSET_BITS  = 16;
  static constexpr u32 ITEMS_OFFSET_MASK  = (1 << ITEMS_OFFSET_BITS) - 1;
  static constexpr u32 ITEMS_OFFSET_SHIFT = 0u;

  static constexpr u32 ITEMS_FILTER_BITS  = 8;
  static constexpr u32 ITEMS_FILTER_MASK  = (1 << ITEMS_FILTER_BITS) - 1;
  static constexpr u32 ITEMS_FILTER_SHIFT = ITEMS_OFFSET_SHIFT + ITEMS_OFFSET_BITS;

  /*static bool node_is_leaf(u32 key) { return 0 != (key & LEAF_BIT); }
  static u32  node_get_num_items(u32 key) { return ((key >> ITEMS_CNT_OFFSET) & ITEMS_CNT_MASK); }
  static u32  node_get_items_offset(u32 key) {
    return ((key >> ITEMS_INDEX_OFFSET) & ITEMS_INDEX_MASK);
  }
  static u32 node_get_num_children(u32 key) { return ((key >> CHILD_CNT_OFFSET) & CHILD_CNT_MASK); }
  static u32 node_get_children_offset(u32 key) {
    return ((key >> CHILD_INDEX_OFFSET) & CHILD_INDEX_MASK);
  }*/

  union {
    u32       leafs[8];
    SDF_Node *child_nodes[8];
  };
  bool is_leaf;
  SDF_Node() { MEMZERO(*this); }
  ~SDF_Node() {
    if (is_leaf) {
      // ito(8) if (leafs[i]) delete leafs[i];
    } else {
      ito(8) if (child_nodes[i]) delete child_nodes[i];
    }
    MEMZERO(*this);
  }

  void init_node(i32 dim, Array<SDF_float8x8x8> &sdf_full, int3 size, SDF_float8x8x8_minmax *blocks,
                 i32 x, i32 y, i32 z) {
    // static int indent = 0;
    if (x - dim / 2 >= size.x || y - dim / 2 >= size.y || z - dim / 2 >= size.z) return;
    i32 step = dim >> 1;
    /* ito(indent) fprintf(stdout, " ");
     fprintf(stdout, "visiting %i %i %i\n", x, y, z);
     indent++;
     defer(--indent);*/
    if (dim == 2) {
      is_leaf = true;
      ito(8) {
        i32 ix = ((i >> 0) & 1) - 1;
        i32 iy = ((i >> 1) & 1) - 1;
        i32 iz = ((i >> 2) & 1) - 1;
        ix     = x + ix;
        iy     = y + iy;
        iz     = z + iz;
        ASSERT_ALWAYS(ix >= 0);
        ASSERT_ALWAYS(iy >= 0);
        ASSERT_ALWAYS(iz >= 0);
        if (ix < size.x && iy < size.y && iz < size.z) {
          /* ito(indent) fprintf(stdout, " ");
           fprintf(stdout, "visiting low level %i %i %i\n", ix, iy, iz);
           fflush(stdout);
           ASSERT_ALWAYS(index_set.contains(ix + iy * size.x + iz * size.x * size.y) == false);
           index_set.insert(ix + iy * size.x + iz * size.x * size.y);*/
          auto &block = blocks[ix + iy * size.x + iz * size.x * size.y];

          if (block.is_surface()) {
            leafs[i] = sdf_full.size + 1;
            SDF_float8x8x8 nb;
            jto(8 * 8 * 8) {
              nb.sdf[j] = normalize_sdf(block.sdf[j], block.aabb.max.x - block.aabb.min.x);
            }
            sdf_full.push(nb);
            // memcpy(&leafs[i]->sdf[0], &block.sdf[0], sizeof(SDF_float8x8x8));
          }
        }
      }
    } else {
      is_leaf = false;
      ito(8) {
        i32 ix = ((i >> 0) & 1) * 2 - 1;
        i32 iy = ((i >> 1) & 1) * 2 - 1;
        i32 iz = ((i >> 2) & 1) * 2 - 1;

        child_nodes[i] = new SDF_Node();
        child_nodes[i]->init_node(step, sdf_full, size, blocks, x + (step * ix) / 2,
                                  y + (step * iy) / 2, z + (step * iz) / 2);
      }
    }
  }
  i32 clean_up() {
    if (is_leaf) {
      i32 cnt = 0;
      ito(8) {
        if (leafs[i] != NULL) //
          cnt++;
      }
      return cnt;
    } else {
      i32 cnt = 0;
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        i32 icnt = child_nodes[i]->clean_up();
        if (icnt == 0) {
          delete child_nodes[i];
          child_nodes[i] = NULL;
        } else {
          cnt += icnt;
        }
      }
      return cnt;
    }
  }
  void render(Gizmo_Layer *gl, float3 const &min, float3 const &max) {
    if (g_config.get_bool("render_tree_interim"))
      gl->render_linebox(min, max, float3(0.0f, 1.0f, 0.0f));
    float3 size   = (max - min) / 2.0f;
    float3 center = min + size;
    if (is_leaf) {
      ito(8) {
        if (leafs[i] == NULL) continue;
        i32    ix         = ((i >> 0) & 1) * 2 - 1;
        i32    iy         = ((i >> 1) & 1) * 2 - 1;
        i32    iz         = ((i >> 2) & 1) * 2 - 1;
        float3 new_size   = size / 2.0f;
        float3 new_center = center + float3(f32(ix), f32(iy), f32(iz)) * new_size;
        gl->render_linebox(new_center - new_size, new_center + new_size, float3(1.0f, 0.0f, 0.0f));
      }
    } else {
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        i32    ix         = ((i >> 0) & 1) * 2 - 1;
        i32    iy         = ((i >> 1) & 1) * 2 - 1;
        i32    iz         = ((i >> 2) & 1) * 2 - 1;
        float3 new_size   = size / 2.0f;
        float3 new_center = center + float3(f32(ix), f32(iy), f32(iz)) * new_size;
        child_nodes[i]->render(gl, new_center - new_size, new_center + new_size);
      }
    }
  }

  void pack(Array<SDF_float8x8x8> &sdf_full, Array<u32> &nodes, Array<SDF_u168x8x8> &sdf_pack) {
    if (is_leaf) {
      u32 key    = LEAF_BIT;
      u32 offset = sdf_pack.size;
      ASSERT_ALWAYS(offset < (1 << 31));
      key      = key | ((offset & ITEMS_OFFSET_MASK) << ITEMS_OFFSET_SHIFT);
      u32 mask = 0;
      ito(8) {
        if (leafs[i] == NULL) continue;
        mask |= (1 << 0);
        SDF_u168x8x8 hv;
        jto(8 * 8 * 8) { hv.sdf[j] = pack_sdf(sdf_full[leafs[i] - 1].sdf[j]); }
        sdf_pack.push(hv);
      }
      key = key | ((mask & ITEMS_FILTER_MASK) << ITEMS_FILTER_SHIFT);
      nodes.push(key);
    } else {
      u32 key        = 0;
      u32 mask       = 0;
      u32 offsets[8] = {};
      u32 offset     = nodes.size;
      nodes.push(0);
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        nodes.push(0);
      }
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        mask |= (1 << 0);
        offsets[i] = nodes.size;
        child_nodes[i]->pack(sdf_full, nodes, sdf_pack);
      }
      key           = key | ((mask & ITEMS_FILTER_MASK) << ITEMS_FILTER_SHIFT);
      nodes[offset] = key;
      u32 cnt       = 0;
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        nodes[offset + 1 + cnt++] = offsets[i];
      }
    }
  }

  template <typename F> void traverse(float3 const &min, float3 const &max, F f) {
    float3 size   = (max - min) / 2.0f;
    float3 center = min + size;
    if (is_leaf) {
      ito(8) {
        if (leafs[i] == NULL) continue;
        i32    ix         = ((i >> 0) & 1) * 2 - 1;
        i32    iy         = ((i >> 1) & 1) * 2 - 1;
        i32    iz         = ((i >> 2) & 1) * 2 - 1;
        float3 new_size   = size / 2.0f;
        float3 new_center = center + float3(f32(ix), f32(iy), f32(iz)) * new_size;
        f(new_center - new_size, new_center + new_size, leafs[i] - 1);
      }
    } else {
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        i32    ix         = ((i >> 0) & 1) * 2 - 1;
        i32    iy         = ((i >> 1) & 1) * 2 - 1;
        i32    iz         = ((i >> 2) & 1) * 2 - 1;
        float3 new_size   = size / 2.0f;
        float3 new_center = center + float3(f32(ix), f32(iy), f32(iz)) * new_size;
        child_nodes[i]->traverse(new_center - new_size, new_center + new_size, f);
      }
    }
  }
  struct Collision {
    float3 position;
    float  near_t;
    float  far_t;
    int    node;
  };
  bool getIntersection(SDF_float8x8x8 *sdfs, float3 const &min, float3 const &max, Ray const &ray,
                       Collision &out) {
    Collision tmp = out;
    if (!AABB{min, max}.collide(ray.o, ray.d, tmp.near_t, tmp.far_t)) {
      return false;
    }
    if (g_config.get_bool("viz_tree_traversal"))
      gizmo_layer->render_linebox(min, max, float3(1.0f, 1.0f, 0.0f));
    float3 size   = (max - min) / 2.0f;
    float3 center = min + size;
    if (is_leaf) {
      bool found = false;
      ito(8) {
        if (leafs[i] == NULL) continue;

        i32    ix         = ((i >> 0) & 1) * 2 - 1;
        i32    iy         = ((i >> 1) & 1) * 2 - 1;
        i32    iz         = ((i >> 2) & 1) * 2 - 1;
        float3 new_size   = size / 2.0f;
        float3 new_center = center + float3(f32(ix), f32(iy), f32(iz)) * new_size;
        if (g_config.get_bool("viz_tree_traversal"))
          gizmo_layer->render_linebox(min, max, float3(1.0f, 0.0f, 0.0f));
        Collision tmp = out;
        if (AABB{new_center - new_size, new_center + new_size}.collide(ray.o, ray.d, tmp.near_t,
                                                                       tmp.far_t)) {
          if (g_config.get_bool("viz_raymarch_aabb_colisions")) {
            gizmo_layer->draw_ss_circle(ray.o + ray.d * tmp.near_t, 0.01f,
                                        float3(0.0f, 0.0f, 1.0f));
            gizmo_layer->draw_ss_circle(ray.o + ray.d * tmp.far_t, 0.02f, float3(0.0f, 0.5f, 1.0f));
          }
          float3 wro   = ray.o + ray.d * tmp.near_t;
          float3 oro   = wro - new_center;
          float  scale = size.x;
          oro /= new_size.x;
          /*oro += float3(1.0f / 16.0f, 1.0f / 16.0f, 1.0f / 16.0f);
          oro *= 1.0f - 1.0f / 8.0f;*/
          float3 rd   = ray.d;
          float  dist = 1.0e6f;
          if (g_config.get_bool("viz_raymarch_probes"))
            sdfs[leafs[i] - 1].render_viz(new_center - new_size, new_center + new_size);
          if (g_config.get_bool("viz_raymarch_probes_values"))
            sdfs[leafs[i] - 1].render_values_viz(new_center - new_size, new_center + new_size);
          f32   margin      = g_config.get_f32("raymarch_margin", 1.0e-3f, 0.0f, 1.0e-1f);
          float t           = tmp.near_t + 1.0e-3f;
          bool  local_found = false;
          // ray march to find intersection
          jto(g_config.get_u32("cpu_sdf_iterations")) {
            // if (oro.x > 1.0f || oro.x < -1.0f || oro.y > 1.0f || oro.y < -1.0f || oro.z > 1.0f ||
            //    oro.z < -1.0f) {
            //  dist = 1.0f;
            //  break;
            //}

            if (g_config.get_bool("viz_raymarch_interpolate_debug"))
              dist = sdfs[leafs[i] - 1].sample_debug(new_center - new_size, new_center + new_size,
                                                     wro, oro);
            else
              dist = sdfs[leafs[i] - 1].sample(oro);
            dist *= (1.0f + g_config.get_f32("raymarch_correction", 0.0f, -1.0f, 1.0f));
            if (g_config.get_bool("viz_raymarch_circles"))
              gizmo_layer->draw_ss_circle(wro, dist * scale, float3(0.0f, 1.0f, 0.0f));
            if (g_config.get_bool("viz_raymarch_probes_values"))
              gizmo_layer->draw_string(wro, float3(0.0f, 1.0f, 0.0f), "%f", dist);
            if (abs(dist) < margin) {
              local_found = true;
              break;
            }
            t += dist * scale;
            if (t <= tmp.near_t || t >= tmp.far_t) {
              local_found = false;
              break;
            }
            oro += rd * dist * 2.0f;
            wro += rd * dist * scale;
          }
          if (local_found) {
            // out.near_t   = t;
            // out.far_t    = t;
            out.position = wro;
            out.node     = leafs[i] - 1;
            found        = true;
          }
        }
      }
      return found;
    } else {
      bool found = false;
      ito(8) {
        if (child_nodes[i] == NULL) continue;
        i32    ix         = ((i >> 0) & 1) * 2 - 1;
        i32    iy         = ((i >> 1) & 1) * 2 - 1;
        i32    iz         = ((i >> 2) & 1) * 2 - 1;
        float3 new_size   = size / 2.0f;
        float3 new_center = center + float3(f32(ix), f32(iy), f32(iz)) * new_size;
        if (child_nodes[i]->getIntersection(sdfs, new_center - new_size, new_center + new_size, ray,
                                            out)) {
          found = true;
        }
      }
      return found;
    }
  }
};

struct SDF_Root_Node {
  SDF_Node *root = NULL;
  i32       dim;
  i32       depth;
  i32       num_leaf_blocks;
  i32       num_interim_nodes;
  AABB      aabb;
  AABB      real_aabb;

  Array<SDF_float8x8x8> sdf_full = {};
  Array<u32>            nodes    = {};
  Array<SDF_u168x8x8>   sdf      = {};

  void render(Gizmo_Layer *gl) {
    gl->render_linebox(real_aabb.min, real_aabb.max, float3(0.0f, 0.0f, 1.0f));
    root->render(gl, aabb.min, aabb.max);
  }

  void dump(string_ref path) {
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "wb");
    fwrite(&dim, 1, sizeof(dim), f);
    fwrite(&depth, 1, sizeof(depth), f);
    fwrite(&num_leaf_blocks, 1, sizeof(num_leaf_blocks), f);
    fwrite(&num_interim_nodes, 1, sizeof(num_interim_nodes), f);
    fwrite(&aabb, 1, sizeof(aabb), f);
    fwrite(&real_aabb, 1, sizeof(real_aabb), f);

    fwrite(&nodes[0], 1, sizeof(nodes[0]) * nodes.size, f);
    fwrite(&sdf[0], 1, sizeof(sdf[0]) * sdf.size, f);
    fclose(f);
  }

  void pack() {
    nodes.release();
    sdf.release();
    root->pack(sdf_full, nodes, sdf);
  }

  template <typename F> void traverse(F f) { root->traverse(aabb.min, aabb.max, f); }

  ~SDF_Root_Node() {
    nodes.release();
    sdf.release();
    if (root) delete root;
    MEMZERO(*this);
  }

  bool getIntersection(Ray const &ray, SDF_Node::Collision &out) {
    out.far_t  = 1.0e6f;
    out.near_t = 1.0e6f;
    if (!aabb.collide(ray.o, ray.d, out.near_t, out.far_t)) {
      return false;
    }
    return root->getIntersection(&sdf_full[0], aabb.min, aabb.max, ray, out);
  }
};

struct SDF_vfx8_mc {
  struct GFX_Cache {
    Resource_ID vbo_position;
    Resource_ID vbo_normal;
    Resource_ID ibo;
    i32         num_indices;
  };
  GFX_Cache *gfx_cache = NULL;

  Mesh *meshes       = NULL;
  i32   num_meshes   = 0;
  i32   num_vertices = 0;
  i32   num_indices  = 0;

  void release() {
    if (meshes) {
      ito(num_meshes) {
        delete[] meshes[i].vertices;
        delete[] meshes[i].faces;
      }
      delete[] meshes;
    }
    MEMZERO(*this);
  }

  void release_gfx(rd::IFactory *factory) {
    if (gfx_cache) {
      factory->release_resource(gfx_cache->vbo_position);
      factory->release_resource(gfx_cache->vbo_normal);
      factory->release_resource(gfx_cache->ibo);
      delete gfx_cache;
      gfx_cache = NULL;
    }
  }

  void render_meshes(rd::IFactory *factory, rd::Imm_Ctx *ctx) {
    // if (mesh.vertexCount)
    {
      ctx->push_state();
      defer(ctx->pop_state());
      // ctx->clear_state();
      // setup_default_state(ctx, 1);
      ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
)

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_NORMAL))

@(ENTRY)
  PIXEL_NORMAL   = NORMAL;
  @(EXPORT_POSITION
      mul4(viewproj, float4(POSITION, 1.0))
  );
@(END)
)"),
                                                    NULL, 0));

      ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_NORMAL))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  @(EXPORT_COLOR 0 abs(float4(PIXEL_NORMAL, 1.0)));
@(END)
)"),
                                                    NULL, 0));
      if (gfx_cache == NULL) {
        gfx_cache = new GFX_Cache;

        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits       = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits     = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
          buf_info.size           = sizeof(float3) * num_vertices;
          gfx_cache->vbo_position = factory->create_buffer(buf_info);
          {
            float3 *vertices = (float3 *)factory->map_buffer(gfx_cache->vbo_position);
            i32     cursor   = 0;
            ito(num_meshes) {
              memcpy(vertices + cursor, meshes[i].vertices, sizeof(float3) * meshes[i].vertexCount);
              cursor += meshes[i].vertexCount;
            }
            factory->unmap_buffer(gfx_cache->vbo_position);
          }
        }
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits     = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits   = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
          buf_info.size         = sizeof(float3) * num_vertices;
          gfx_cache->vbo_normal = factory->create_buffer(buf_info);
          {
            float3 *vertices = (float3 *)factory->map_buffer(gfx_cache->vbo_normal);
            i32     cursor   = 0;
            ito(num_meshes) {
              memcpy(vertices + cursor, meshes[i].normals, sizeof(float3) * meshes[i].vertexCount);
              cursor += meshes[i].vertexCount;
            }
            factory->unmap_buffer(gfx_cache->vbo_normal);
          }
        }
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
          buf_info.size       = num_indices * sizeof(u32);
          gfx_cache->ibo      = factory->create_buffer(buf_info);
          {
            u32 *indices     = (u32 *)factory->map_buffer(gfx_cache->ibo);
            i32  cursor      = 0;
            i32  mesh_offset = 0;
            ito(num_meshes) {
              jto(meshes[i].faceCount * 3) {
                indices[cursor++] = meshes[i].faces[j] + mesh_offset;
                // memcpy(indices + cursor, meshes[i].normals, sizeof(float3) *
                // meshes[i].vertexCount);
              }
              mesh_offset += meshes[i].vertexCount;
            }
            // memcpy(indices, mesh.faces, mesh.faceCount * 3 * sizeof(u32));
            factory->unmap_buffer(gfx_cache->ibo);
          }
        }
        gfx_cache->num_indices = num_indices;
      }
      ctx->IA_set_vertex_buffer(0, gfx_cache->vbo_position, 0, 12, rd::Input_Rate::VERTEX);
      ctx->IA_set_vertex_buffer(1, gfx_cache->vbo_normal, 0, 12, rd::Input_Rate::VERTEX);
      ctx->IA_set_index_buffer(gfx_cache->ibo, 0, rd::Index_t::UINT32);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        ctx->IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 1;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 1;
        info.offset   = 0;
        info.type     = rd::Attriute_t::NORMAL;
        ctx->IA_set_attribute(info);
      }
      struct PC {
        // float4x4 world_transform;
        float4x4 viewproj;
      } pc;
      // pc.world_transform = model;
      pc.viewproj = gizmo_layer->get_camera().viewproj();
      ctx->push_constants(&pc, 0, sizeof(pc));
      ctx->draw_indexed(gfx_cache->num_indices, 1, 0, 0, 0);
      if (g_config.get_bool("render_marched_wireframe")) {
        rd::RS_State rs_state;
        MEMZERO(rs_state);
        rs_state.polygon_mode = rd::Polygon_Mode::LINE;
        rs_state.front_face   = rd::Front_Face::CW;
        rs_state.cull_mode    = rd::Cull_Mode::NONE;
        ctx->RS_set_state(rs_state);
        ctx->RS_set_line_width(2.2f);
        ctx->RS_set_depth_bias(-30.0f);
        ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_NORMAL))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  @(EXPORT_COLOR 0 float4_splat(0.0));
@(END)
)"),
                                                      NULL, 0));
        ctx->draw_indexed(gfx_cache->num_indices, 1, 0, 0, 0);
      }
    }
  }
  ~SDF_vfx8_mc() { release(); }
};

struct SDF_float8x8x8_Volume {
  SDF_float8x8x8_minmax *blocks = NULL;
  int3                   size;
  float3                 min, max;
  float                  voxel_size;
  float                  block_size;
  i32                    num_boundary_blocks = 0;

  void release() {
    if (blocks) delete[] blocks;
    MEMZERO(*this);
  }
  ~SDF_float8x8x8_Volume() { release(); }
  SDF_Root_Node *generate_tree() {
    i32 max_dim = MAX3(size.x, size.y, size.z);
    i32 pot     = (i32)smallest_pot((u32)max_dim);
    i32 dim     = pot;
    i32 depth   = 0;
    if (pot < 2) pot = 2;
    while (pot) {
      pot = pot >> 1;
      depth++;
    }
    depth--;
    SDF_Root_Node *root = new SDF_Root_Node;
    root->dim           = dim;
    root->depth         = depth;
    root->aabb          = {min, min + float(dim) * float3(block_size, block_size, block_size)};
    root->real_aabb     = {min, max};
    root->root          = new SDF_Node;
    // index_set.init();
    root->root->init_node(dim, root->sdf_full, size, blocks, dim / 2, dim / 2, dim / 2);
    // ASSERT_ALWAYS(index_set.item_count == size.x * size.y * size.z);
    root->num_interim_nodes = root->root->clean_up() - num_boundary_blocks;
    root->num_leaf_blocks   = num_boundary_blocks;
    return root;
  }
  void dump(string_ref path) {
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "wb");
    fwrite(&size, 1, sizeof(size), f);
    fwrite(&min, 1, sizeof(min), f);
    fwrite(&max, 1, sizeof(max), f);
    fwrite(&voxel_size, 1, sizeof(voxel_size), f);
    fwrite(&block_size, 1, sizeof(block_size), f);
    fwrite(&num_boundary_blocks, 1, sizeof(num_boundary_blocks), f);
    fwrite(blocks, 1, sizeof(SDF_float8x8x8_minmax) * size.x * size.y * size.z, f);
    fclose(f);
    {
      FILE *f = fopen(stref_to_tmp_cstr(stref_concat_tmp(stref_substring(path, stref_s(".bin")),
                                                         stref_s("dense.bin"))),
                      "wb");
      fwrite(&size, 1, sizeof(size), f);
      fwrite(&min, 1, sizeof(min), f);
      fwrite(&max, 1, sizeof(max), f);
      fwrite(&voxel_size, 1, sizeof(voxel_size), f);
      fwrite(&block_size, 1, sizeof(block_size), f);
      fwrite(&num_boundary_blocks, 1, sizeof(num_boundary_blocks), f);
      zto(size.z) {
        yto(size.y) {
          xto(size.x) {
            auto &block = blocks[x + y * size.x + z * size.x * size.y];
            if (block.min_val < 0.0f && block.max_val > 0.0f) {
              fwrite(&block, 1, sizeof(SDF_float8x8x8_minmax), f);
            }
          }
        }
      }

      fclose(f);
    }
  }
  bool restore(string_ref path) {
    release();
    TMP_STORAGE_SCOPE;
    FILE *f = fopen(stref_to_tmp_cstr(path), "rb");
    if (f == NULL) return false;
    fread(&size, 1, sizeof(size), f);
    if (size.x * size.y * size.z <= 0) return false;
    fread(&min, 1, sizeof(min), f);
    fread(&max, 1, sizeof(max), f);
    fread(&voxel_size, 1, sizeof(voxel_size), f);
    fread(&block_size, 1, sizeof(block_size), f);
    fread(&num_boundary_blocks, 1, sizeof(num_boundary_blocks), f);
    blocks = new SDF_float8x8x8_minmax[size.x * size.y * size.z];
    fread(blocks, 1, sizeof(SDF_float8x8x8_minmax) * size.x * size.y * size.z, f);
    fclose(f);
    return true;
  }
  SDF_vfx8_mc *march() {
    if (num_boundary_blocks <= 0) return NULL;
    SDF_vfx8_mc *out = new SDF_vfx8_mc;
    out->meshes      = new Mesh[num_boundary_blocks];
    zto(size.z) {
      yto(size.y) {
        xto(size.x) {
          auto &block = blocks[x + y * size.x + z * size.x * size.y];
          if (block.is_surface()) {
            i32 mesh_id          = out->num_meshes++;
            out->meshes[mesh_id] = ::march(block.sdf, 8, 8, 8, 0.0f);
            /*out->meshes[mesh_id].offset[0] = block.aabb.min.x;
            out->meshes[mesh_id].offset[1] = block.aabb.min.y;
            out->meshes[mesh_id].offset[2] = block.aabb.min.z;*/
            auto &mesh = out->meshes[mesh_id];
            ito(mesh.vertexCount) {
              mesh.vertices[i][0] =
                  block.aabb.min.x +
                  (block.aabb.max.x - block.aabb.min.x + voxel_size) * mesh.vertices[i][0] / 8.0f;
              mesh.vertices[i][1] =
                  block.aabb.min.y +
                  (block.aabb.max.y - block.aabb.min.y + voxel_size) * mesh.vertices[i][1] / 8.0f;
              mesh.vertices[i][2] =
                  block.aabb.min.z +
                  (block.aabb.max.z - block.aabb.min.z + voxel_size) * mesh.vertices[i][2] / 8.0f;
            }
            out->num_vertices += out->meshes[mesh_id].vertexCount;
            out->num_indices += out->meshes[mesh_id].faceCount * 3;
          }
        }
      }
    }
    return out;
  }
};

struct CPU_ONode {
  AABB aabb;
  union {
    struct {
      CPU_ONode *children[2 * 2 * 2];
    };
    u32 block_id;
  };
  bool is_leaf;
  CPU_ONode() { MEMZERO(*this); }
  ~CPU_ONode() {
    if (is_leaf == false) {
      ito(2 * 2 * 2) delete children[i];
    }
  }
};

struct GPU_SDF_float8x8x8_Volume {
  struct GPU_SDF_float8x8x8 {
    AABB        aabb;
    Resource_ID volume_texture;
  };
  Array<GPU_SDF_float8x8x8> resources;
  void                      init(rd::IFactory *f, SDF_float8x8x8_Volume *blocks) {
    resources.init();
    Resource_ID staging_buf{};
    defer(f->release_resource(staging_buf));
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      buf_info.size       = 8 * 8 * 8 * sizeof(f32);
      staging_buf         = f->create_buffer(buf_info);
    }
    zto(blocks->size.z) {
      yto(blocks->size.y) {
        xto(blocks->size.x) {
          auto &block =
              blocks->blocks[x + y * blocks->size.x + z * blocks->size.x * blocks->size.y];
          if (block.min_val < 0.0f && block.max_val > 0.0f) {
            rd::Image_Create_Info info;
            MEMZERO(info);
            info.width      = 8;
            info.height     = 8;
            info.depth      = 8;
            info.layers     = 1;
            info.levels     = 1;
            info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
            info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                              (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
            info.format        = rd::Format::R32_FLOAT;
            Resource_ID img_id = f->create_image(info);
            f32 *       ptr    = (f32 *)f->map_buffer(staging_buf);
            memcpy(ptr, block.sdf, sizeof(f32) * 8 * 8 * 8);
            f->unmap_buffer(staging_buf);
            auto ctx = f->start_compute_pass();
            ctx->image_barrier(img_id, (u32)rd::Access_Bits::MEMORY_WRITE,
                               rd::Image_Layout::TRANSFER_DST_OPTIMAL);
            ctx->copy_buffer_to_image(staging_buf, 0, img_id, rd::Image_Copy::top_level());
            f->end_compute_pass(ctx);
            GPU_SDF_float8x8x8 b;
            b.aabb           = block.aabb;
            b.volume_texture = img_id;
            resources.push(b);
          }
        }
      }
    }
  }
  void release(rd::IFactory *f) {
    ito(resources.size) f->release_resource(resources[i].volume_texture);
    resources.release();
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

  SDF_float8x8x8_Volume *generate_blocks() {
    ASSERT_ALWAYS(((size.x - 1) % 7) == 0);
    ASSERT_ALWAYS(((size.y - 1) % 7) == 0);
    ASSERT_ALWAYS(((size.z - 1) % 7) == 0);
    //
    //  8x8x8 blocks define 7x7x7 cells
    //  boundary needs to be duplicated for neighbors
    //
    // TEXELS:
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    // | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * | * |
    // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    //
    // CELLS:
    //   |                           |
    //   V                           V
    //   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //  -*---*---*---*---*---*---*---*---*---*---*---*---*---*---*
    //   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    //
    // +-------------------------------+
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // | | * | * | * | * | * | * | * | |
    // | +---+---+---+---+---+---+---+ |
    // +-------------------------------+
    //
    int3                   new_size = int3((size.x - 1) / 7, (size.y - 1) / 7, (size.z - 1) / 7);
    SDF_float8x8x8_Volume *out      = new SDF_float8x8x8_Volume;
    out->size                       = new_size;
    out->blocks     = new SDF_float8x8x8_minmax[new_size.x * new_size.y * new_size.z];
    out->block_size = voxel_size * 7;
    out->min        = min + float3(voxel_size, voxel_size, voxel_size) / 2.0f;
    out->max        = max - float3(voxel_size, voxel_size, voxel_size) / 2.0f;
    out->voxel_size = voxel_size;
    zto(new_size.z) {
      yto(new_size.y) {
        xto(new_size.x) {
          SDF_float8x8x8_minmax block;
          i32                   offset_x = x * 7;
          i32                   offset_y = y * 7;
          i32                   offset_z = z * 7;
          block.max_val                  = -std::numeric_limits<float>::infinity();
          block.min_val                  = std::numeric_limits<float>::infinity();
          block.aabb.min = float3(offset_x * out->voxel_size, offset_y * out->voxel_size,
                                  offset_z * out->voxel_size);
          block.aabb.max =
              block.aabb.min + 7.0f * float3(out->voxel_size, out->voxel_size, out->voxel_size);
          block.voxel_size = out->voxel_size;
          ito(8) {
            jto(8) {
              kto(8) {
                i32 ix  = offset_x + k;
                i32 iy  = offset_y + j;
                i32 iz  = offset_z + i;
                f32 val = sdf[ix + iy * size.x + iz * size.x * size.y];
                // if (val < 0.0f) DebugBreak();
                block.max_val = MAX(val, block.max_val);
                block.min_val = MIN(val, block.min_val);
                block.avg_val += val;
                block.sdf[k + j * 8 + i * 8 * 8] = val;
              }
            }
          }
          block.avg_val /= (8 * 8 * 8);
          if (block.min_val < 0.0f && block.max_val > 0.0f) out->num_boundary_blocks++;
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
  std::thread *     thread = NULL;
  std::atomic<bool> ready;

  Raw_SDF_Volume *       volume        = NULL;
  SDF_float8x8x8_Volume *volume_blocks = NULL;
  SDF_vfx8_mc *          mcs           = NULL;
  // GPU_SDF_float8x8x8_Volume *gpu_blocks    = NULL;
  // CPU_ONode *                oroot         = NULL;
  SDF_Root_Node *root = NULL;
  void           release() {
    if (volume) delete volume;
    if (volume_blocks) delete volume_blocks;
    if (mcs) delete mcs;
    if (root) delete root;
    // delete oroot;
    // delete gpu_blocks;
    MEMZERO(*this);
  }
  bool isReady() {
    if (!ready) return false;
    if (thread) {
      thread->join();
      delete thread;
      thread = NULL;
    }
    return true;
  }
  void load(string_ref path) {
    thread = new std::thread([this, path] {
      volume = new Raw_SDF_Volume;
      if (!volume->restore(stref_s("raw_sdf.bin"))) {

        TMP_STORAGE_SCOPE;
        FILE *f = fopen(stref_to_tmp_cstr(path), "rb");
        ASSERT_ALWAYS(f);
        // w h d
        // min.x min.y min.z
        // voxel_size
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

        volume->sdf        = sdf;
        volume->size       = int3(w, h, d);
        volume->min        = float3(minx, miny, minz);
        volume->max        = float3(minx + dr * w, miny + dr * h, minz + dr * d);
        volume->voxel_size = dr;
        fprintf(stdout, "SDF has been parsed\n");
        volume->dump(stref_s("raw_sdf.bin"));
        fprintf(stdout, "SDF has been dumped\n");
      } else {
        fprintf(stdout, "SDF has been restored from cache\n");
      }
      if (!volume_blocks) {
        volume_blocks = new SDF_float8x8x8_Volume;
        if (!volume_blocks->restore(stref_s("blocks_sdf.bin"))) {
          delete volume_blocks;
          volume_blocks = volume->generate_blocks();
          fprintf(stdout, "SDF bocks have been generated\n");
          volume_blocks->dump(stref_s("blocks_sdf.bin"));
          fprintf(stdout, "SDF Blocks have been dumped\n");
        } else {
          fprintf(stdout, "SDF Blocks have been restored from cache\n");
        }
      }
      if (volume_blocks) {
        fprintf(stdout, "Generating tree...\n");
        root = volume_blocks->generate_tree();
        fprintf(stdout, "Done generating tree.\n");
        {
          fprintf(stdout, "Generating a pack...\n");

          root->pack();
          root->dump(stref_s("fp16_pack.bin"));
          fprintf(stdout, "Done generating a pack.\n");
        }

        fprintf(stdout, "Generating isosurfaces...\n");
        mcs = volume_blocks->march();
        fprintf(stdout, "Done generating isosurfaces.\n");
      }

      { ready = true; }
    });
  }
} g_sdf_loader;

class GBufferPass {
  public:
  Resource_ID normal_rt;
  Resource_ID depth_rt;

  struct GPU_Cube {
    Resource_ID vertex_buffer = {};
    Resource_ID index_buffer  = {};
    void        init(rd::IFactory *factory) {
      if (vertex_buffer.is_null()) {

        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER |
                              (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        buf_info.size = sizeof(float3) * 8;
        vertex_buffer = factory->create_buffer(buf_info);

        float3 dvertices[8] = {float3(-1, -1, -1), float3(1, -1, -1), float3(1, 1, -1),
                               float3(-1, 1, -1),  float3(-1, -1, 1), float3(1, -1, 1),
                               float3(1, 1, 1),    float3(-1, 1, 1)};
        init_buffer(factory, vertex_buffer, dvertices, sizeof(dvertices));
      }
      if (index_buffer.is_null()) {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER |
                              (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        buf_info.size           = 6 * 2 * 3 * sizeof(u32);
        index_buffer            = factory->create_buffer(buf_info);
        u32 dindices[6 * 2 * 3] = {0, 1, 3, 3, 1, 2, 1, 5, 2, 2, 5, 6, 5, 4, 6, 6, 4, 7,
                                   4, 0, 7, 7, 0, 3, 3, 2, 7, 7, 2, 6, 4, 5, 0, 0, 5, 1};
        init_buffer(factory, index_buffer, dindices, sizeof(dindices));
      }
    }
    void bind(rd::Imm_Ctx *ctx) {
      ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, 12, rd::Input_Rate::VERTEX);
      ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        ctx->IA_set_attribute(info);
      }
    }
    void draw(rd::Imm_Ctx *ctx) { ctx->draw_indexed(36, 1, 0, 0, 0); }
    void release(rd::IFactory *factory) {
      if (vertex_buffer.is_null() == false) factory->release_resource(vertex_buffer);
      if (index_buffer.is_null() == false) factory->release_resource(index_buffer);
    }
  };

  struct GPU_SDF {
    SDF_Root_Node *    root            = NULL;
    Resource_ID        texture_sampler = {};
    Resource_ID        nodes           = {};
    Array<Resource_ID> sdf             = {};
    GPU_Cube           gfx_cube        = {};
    Resource_ID        instance_buffer = {};

    struct PC {
      float3 offset;
      float  scale;
      i32    texture_index;
    };

    void release(rd::IFactory *factory) {
      if (nodes.is_null() == false) factory->release_resource(nodes);
      if (texture_sampler.is_null() == false) factory->release_resource(texture_sampler);
      if (instance_buffer.is_null() == false) factory->release_resource(instance_buffer);
      ito(sdf.size) {
        if (sdf[i].is_null() == false) factory->release_resource(sdf[i]);
      }
      nodes = {};
      sdf.release();
      gfx_cube.release(factory);
    }
    void render(rd::IFactory *factory, rd::Imm_Ctx *ctx) {
      gfx_cube.bind(ctx);
      ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
struct Instance_Params {
  float3 offset;
  float scale;
  int texture_index;
};

@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type Instance_Params)
  (name params)
)

@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3)    (name camera_pos))
  (add_field (type int)       (name mode))
  (add_field (type int)       (name sdf_iterations))
)

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_WORLD_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_OBJECT_POSITION))
@(DECLARE_OUTPUT (location 2) (type "flat int")  (name PIXEL_TEXTURE_ID))
@(DECLARE_OUTPUT (location 3) (type float)  (name PIXEL_SCALE))

@(ENTRY)
  float3 offset = params[INSTANCE_INDEX].offset.xyz;
  float scale   = params[INSTANCE_INDEX].scale;
  PIXEL_WORLD_POSITION    = POSITION * scale + offset;
  PIXEL_OBJECT_POSITION   = POSITION;
  PIXEL_TEXTURE_ID        = params[INSTANCE_INDEX].texture_index;
  PIXEL_SCALE             = (scale * 2.0);
  @(EXPORT_POSITION
     mul4(viewproj, float4(PIXEL_WORLD_POSITION, 1.0))
  );
@(END)
)"),
                                                    NULL, 0));
      ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_IMAGE
  (type SAMPLED)
  (array_size 5000)
  (dim 3D)
  (set 2)
  (binding 0)
  (name sdf_textures)
)
@(DECLARE_SAMPLER
  (set 1)
  (binding 1)
  (name my_sampler)
)
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3)    (name camera_pos))
  (add_field (type int)       (name mode))
  (add_field (type int)       (name sdf_iterations))
)
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_WORLD_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_OBJECT_POSITION))
@(DECLARE_INPUT (location 2) (type "flat int")  (name PIXEL_TEXTURE_ID))
@(DECLARE_INPUT (location 3) (type float)  (name PIXEL_SCALE))

layout(depth_greater) out float gl_FragDepth;

@(DECLARE_RENDER_TARGET  (location 0))

float eval_sdf(float3 p) {
  p = p * 0.5 + float3_splat(0.5);
  p *= float3_splat(1.0 - 1.0 / 8.0);
  p += float3_splat(1.0 / 16.0);
  
  float sdf = textureLod(
              sampler3D(
                sdf_textures[nonuniformEXT(PIXEL_TEXTURE_ID)], my_sampler
              ),
              p, 0.0
            ).x;
  return (sdf * 2.0f - 1.0f) * sqrt(3.0);
}

float3 eval_normal(float3 pos) {
    float3 e = float3(1.0, 0.0, -1.0) * 1.0 / 8.0;
    float3 n = float3_splat(0.0);
    if (pos.x > 0.0)
      n += e.zyy * eval_sdf(pos + e.zyy);
    else
      n += e.xyy * eval_sdf(pos + e.xyy);

    if (pos.y > 0.0)
      n += e.yzy * eval_sdf(pos + e.yzy);
    else
      n += e.yxy * eval_sdf(pos + e.yxy);

    if (pos.z > 0.0)
      n += e.yyz * eval_sdf(pos + e.yyz);
    else
      n += e.yyx * eval_sdf(pos + e.yyx);
    return normalize(n);
    //return normalize(
    //        e.xyy * eval_sdf(pos + e.xyy) + 
				//	  e.yyx * eval_sdf(pos + e.yyx) + 
				//	  e.yxy * eval_sdf(pos + e.yxy)
    //        // e.zyy * eval_sdf(pos + e.zyy) + 
				//	  // e.yyz * eval_sdf(pos + e.yyz) + 
				//	  // e.yzy * eval_sdf(pos + e.yzy)
    //      );
}

@(ENTRY)
  float3 ro = PIXEL_OBJECT_POSITION;
  if (mode == 0) {
    float3 rw = PIXEL_WORLD_POSITION;
    float3 rd = normalize(PIXEL_WORLD_POSITION - camera_pos);
    float dist = 0.0;
    for (int i = 0; i < sdf_iterations; i++) {
      dist = eval_sdf(ro);
      if (abs(dist) < 1.0e-3)
        break;
      // dist *= (1.0 - 1.0e-1);
      ro += dist * rd * 2.0;
      rw += dist * rd * PIXEL_SCALE;
    }
    if (any(greater(abs(ro), float3_splat(1.0))))
      discard;
    if (abs(dist) > 1.0e-2)
      discard;
    float3 normal = eval_normal(ro);
    float4 color = float4(normal, 1.0);
    float4 pp = mul4(viewproj, float4(rw, 1.0));
    pp.z /= pp.w;
    gl_FragDepth = pp.z;
    @(EXPORT_COLOR 0 float4_splat(abs(dot(color.xyz, normalize(float3(1.0, 1.0, 1.0))))));
  } else if (mode == 1) {
    float dist = eval_sdf(ro);
    float4 color =
        float4(1.0, 0.0, 0.0, 1.0) * saturate(dist) +
        float4(0.0, 0.0, 1.0, 1.0) * saturate(-dist);
    @(EXPORT_COLOR 0 color);
    float4 pp = mul4(viewproj, float4(PIXEL_WORLD_POSITION, 1.0));
    pp.z /= pp.w;
    gl_FragDepth = pp.z;
  } else {
    float r = float((PIXEL_TEXTURE_ID >> 0)  & 0xfu) / 15.0;
    float g = float((PIXEL_TEXTURE_ID >> 4)  & 0xfu) / 15.0;
    float b = float((PIXEL_TEXTURE_ID >> 8)  & 0xfu) / 15.0;
    @(EXPORT_COLOR 0 float4(r, g, b, 1.0));
    float4 pp = mul4(viewproj, float4(PIXEL_WORLD_POSITION, 1.0));
    pp.z /= pp.w;
    gl_FragDepth = pp.z;
  }
@(END)
)"),
                                                    NULL, 0));
      struct Uniform {
        afloat4x4 viewproj;
        afloat3   camera_pos;
        uint      mode;
        uint      sdf_iterations;
      };
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits          = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits        = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
        buf_info.size              = sizeof(Uniform);
        Resource_ID uniform_buffer = factory->create_buffer(buf_info);
        factory->release_resource(uniform_buffer);
        Uniform *ptr        = (Uniform *)factory->map_buffer(uniform_buffer);
        ptr->viewproj       = gizmo_layer->get_camera().viewproj();
        ptr->camera_pos     = gizmo_layer->get_camera().pos;
        ptr->mode           = g_config.get_u32("sdf_cube_rendering_mode");
        ptr->sdf_iterations = g_config.get_u32("gpu_sdf_iterations", 16, 1, 128);
        factory->unmap_buffer(uniform_buffer);
        ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
      }

      ctx->bind_storage_buffer(0, 1, instance_buffer, 0, sizeof(PC) * root->num_leaf_blocks);
      ito(sdf.size) {
        ctx->bind_image(2, 0, i, sdf[i], rd::Image_Subresource::top_level(), rd::Format::NATIVE);
      }
      ctx->bind_sampler(1, 1, texture_sampler);
      ctx->draw_indexed(36, root->num_leaf_blocks, 0, 0, 0);
    }
    bool is_ready() { return nodes.is_null() == false; }
    void init(rd::IFactory *factory, SDF_Root_Node *root) {
      if (texture_sampler.is_null()) {
        rd::Sampler_Create_Info info;
        MEMZERO(info);
        info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
        info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
        info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
        info.mag_filter     = rd::Filter::LINEAR;
        info.min_filter     = rd::Filter::NEAREST;
        info.mip_mode       = rd::Filter::NEAREST;
        info.anisotropy     = false;
        info.max_anisotropy = 0.0f;
        texture_sampler     = factory->create_sampler(info);
      }
      gfx_cube.init(factory);
      this->root = root;

      if (instance_buffer.is_null()) {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
        buf_info.size       = sizeof(PC) * root->num_leaf_blocks;
        instance_buffer     = factory->create_buffer(buf_info);
        PC *ptr             = new PC[root->num_leaf_blocks];
        defer(delete[] ptr);
        u32 cnt = 0;
        root->traverse([&](float3 const &min, float3 const &max, u32 index) {
          {
            PC &pc           = ptr[cnt++];
            pc.offset        = min + (max - min) / 2.0f;
            pc.scale         = (max.x - min.x) / 2.0f;
            pc.texture_index = index;
            // pc.world_transform = translate(float4x4(1.0f), min);
            // pc.texture_index   = -1;
            // ctx->push_constants(&pc, 0, sizeof(pc));
            // ctx->draw_indexed(36, 1, 0, 0, 0);
          }
        });
        ASSERT_ALWAYS(root->num_leaf_blocks == cnt);
        init_buffer(factory, instance_buffer, ptr, buf_info.size);
      }
      // renderdoc_ctx.start();
      // defer(renderdoc_ctx.end());
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        buf_info.usage_bits =
            (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        buf_info.size = sizeof(root->nodes[0]) * root->nodes.size;
        nodes         = factory->create_buffer(buf_info);
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
          buf_info.size       = sizeof(root->nodes[0]) * root->nodes.size;
          Resource_ID staging = factory->create_buffer(buf_info);
          factory->release_resource(staging);
          void *data = (void *)factory->map_buffer(staging);
          memcpy(data, &root->nodes[0], sizeof(root->nodes[0]) * root->nodes.size);
          factory->unmap_buffer(staging);
          {
            auto *ctx = factory->start_compute_pass();
            ctx->copy_buffer(staging, 0, nodes, 0, buf_info.size);
            factory->end_compute_pass(ctx);
          }
        }
      }
      {
        Resource_ID staging_buf{};
        defer(factory->release_resource(staging_buf));
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
          buf_info.size       = 8 * 8 * 8 * sizeof(half_float::half);
          staging_buf         = factory->create_buffer(buf_info);
        }
        ito(root->sdf.size) {
          rd::Image_Create_Info info;
          MEMZERO(info);
          info.width      = 8;
          info.height     = 8;
          info.depth      = 8;
          info.layers     = 1;
          info.levels     = 1;
          info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
          info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
          info.format        = rd::Format::R16_UNORM;
          Resource_ID img_id = factory->create_image(info);
          f32 *       ptr    = (f32 *)factory->map_buffer(staging_buf);
          memcpy(ptr, root->sdf[i].sdf, sizeof(u16) * 8 * 8 * 8);
          factory->unmap_buffer(staging_buf);
          auto ctx = factory->start_compute_pass();
          ctx->image_barrier(img_id, (u32)rd::Access_Bits::MEMORY_WRITE,
                             rd::Image_Layout::TRANSFER_DST_OPTIMAL);
          ctx->copy_buffer_to_image(staging_buf, 0, img_id, rd::Image_Copy::top_level());
          factory->end_compute_pass(ctx);
          factory->wait_idle();
          sdf.push(img_id);
        }
      }
    }
  };

  GPU_SDF gfx_sdf = {};

  public:
  TimeStamp_Pool timestamps = {};

  void init() { MEMZERO(*this); }
  void render(rd::IFactory *factory) {
    timestamps.update(factory);
    float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f, 0.0f));
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GfxSufraceComponent>() == NULL) {
          GfxSufraceComponent::create(factory, mn);
        }
        /*  render_bvh(bvh_visualizer_offset, mn->getComponent<GfxSufraceComponent>()->getBVH(),
                     gizmo_layer);*/
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
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 1.0f;
      info.rts.push(rt0);

      info.depth_target.image             = depth_rt;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::NATIVE;

      rd::Imm_Ctx *ctx = factory->start_render_pass(info);
      timestamps.insert(factory, ctx);
      setup_default_state(ctx, 1);
      rd::DS_State ds_state;
      rd::RS_State rs_state;
      float4x4     viewproj = gizmo_layer->get_camera().viewproj();
      ctx->push_constants(&viewproj, 0, sizeof(float4x4));
      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      MEMZERO(ds_state);
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      ctx->DS_set_state(ds_state);

      if (g_sdf_loader.isReady()) {
        if (!gfx_sdf.is_ready()) {
          gfx_sdf.init(factory, g_sdf_loader.root);
        }
        // if (g_sdf_loader.gpu_blocks == NULL) {
        //   g_sdf_loader.gpu_blocks = new GPU_SDF_float8x8x8_Volume;
        //   g_sdf_loader.gpu_blocks->init(factory, g_sdf_loader.volume_blocks);
        // }
        auto &blocks = g_sdf_loader.volume_blocks;
        if (g_config.get_bool("render_block_lines")) {
          gizmo_layer->reserveLines(blocks->size.x * blocks->size.y * blocks->size.z * 8);
          zto(blocks->size.z) {
            yto(blocks->size.y) {
              xto(blocks->size.x) {
                auto &block =
                    blocks->blocks[x + y * blocks->size.x + z * blocks->size.x * blocks->size.y];
                if (block.min_val < 0.0f && block.max_val > 0.0f)
                  gizmo_layer->render_linebox(block.aabb.min, block.aabb.max,
                                              float3(0.0f, 0.0f, 0.0f));
              }
            }
          }
        }
        if (g_sdf_loader.mcs) {
          if (g_config.get_bool("render_marched")) {
            g_sdf_loader.mcs->render_meshes(factory, ctx);
          }
        }
        if (g_sdf_loader.root) {
          if (g_config.get_bool("render_tree")) {
            g_sdf_loader.root->render(gizmo_layer);
          }
        }
      }
      // static bool onetime = true;
      // if (gfx_sdf.is_ready() && onetime) {
      //  renderdoc_ctx.start();
      //}
      // defer(if (gfx_sdf.is_ready() && onetime) {
      //  renderdoc_ctx.end();
      //  onetime = false;
      //});
      if (g_config.get_bool("render_sdf_cubes")) {
        if (gfx_sdf.is_ready()) {
          gfx_sdf.render(factory, ctx);
        }
      }
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
  @(EXPORT_COLOR 0 float4_splat(abs(dot(PIXEL_NORMAL, normalize(float3(1.0, 1.0, 1.0))))));
@(END)
)"),
                                                    NULL, 0));
      static u32 attribute_to_location[] = {
          0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
      };
      if (g_config.get_bool("render_scene")) {
        MEMZERO(ds_state);
        ds_state.cmp_op             = rd::Cmp::GE;
        ds_state.enable_depth_test  = true;
        ds_state.enable_depth_write = true;
        ctx->DS_set_state(ds_state);

        MEMZERO(rs_state);
        rs_state.polygon_mode = rd::Polygon_Mode::FILL;
        rs_state.front_face   = rd::Front_Face::CCW;
        rs_state.cull_mode    = rd::Cull_Mode::BACK;
        ctx->RS_set_state(rs_state);

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
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::LINE;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      ctx->RS_set_state(rs_state);
      ctx->RS_set_depth_bias(0.1f);
      if (g_config.get_bool("render_scene_wireframe")) {
        ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  @(EXPORT_COLOR 0 float4_splat(0.0));
@(END)
)"),
                                                      NULL, 0));
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
      if (g_config.get_bool("render_gizmo")) {
        auto g_camera = gizmo_layer->get_camera();
        {
          float dx = 1.0e-1f * g_camera.distance;
          gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
          gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{dx, 0.0f, 0.0f},
                                     dx * 0.04f, float3{1.0f, 0.0f, 0.0f});
          gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, dx, 0.0f},
                                     dx * 0.04f, float3{0.0f, 1.0f, 0.0f});
          gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, 0.0f, dx},
                                     dx * 0.04f, float3{0.0f, 0.0f, 1.0f});
        }
        gizmo_layer->render(factory, ctx, width, height);
      }
      gizmo_layer->reset();
      timestamps.insert(factory, ctx);
      factory->end_render_pass(ctx);
    }
  }
  void release(rd::IFactory *factory) {
    factory->release_resource(normal_rt);
    // if (g_sdf_loader.gpu_blocks) g_sdf_loader.gpu_blocks->release(factory);
    g_sdf_loader.release();
    gfx_sdf.release(factory);
  }
};

class Event_Consumer : public IGUI_Pass {
  public:
  GBufferPass gbuffer_pass;
  void        init(rd::Pass_Mng *pmng) override { //
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
    ImGui::Text("%fms", gbuffer_pass.timestamps.duration);
    ImGui::End();

    ImGui::Begin("main viewport");
    gizmo_layer->per_imgui_window();
    auto wsize = get_window_size();
    ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                 ImVec2(wsize.x, wsize.y));
    {
      Ray ray = gizmo_layer->getMouseRay();

      SDF_Node::Collision col;
      if (g_sdf_loader.root) {
        if (g_sdf_loader.root->getIntersection(ray, col)) {
          gizmo_layer->draw_sphere(col.position, 1.0e-1f, float3(1.0f, 0.0f, 0.0f));
          if (g_config.get_bool("viz_raymarch"))
            gizmo_layer->draw_line(ray.o, col.position, float3(0.0f, 0.0f, 1.0f));
        }
      }
    }
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
    g_scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
    g_scene->update();
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
    g_sdf_loader.load(stref_s("models/light/source/d5c44c10fd0844fabbb8ddb7a25ad77f.sdf"));
    gbuffer_pass.init();
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
    gbuffer_pass.render(factory);
    IGUI_Pass::on_frame(factory);
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  /* rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
   IGUI_Pass *   gui  = new Event_Consumer;
   gui->init(pmng);
   pmng->set_event_consumer(gui);
   pmng->loop();*/
  return 0;
}
