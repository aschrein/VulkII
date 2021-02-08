#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC

#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "script.hpp"

#include <3rdparty/half.hpp>
#include <atomic>
#include <condition_variable>
#include <imgui.h>
#include <mutex>
#include <thread>

static thread_local u32 g_max_depth = 0;

struct QuadNode {
  private:
  ~QuadNode() = default;
  QuadNode()  = default;

  public:
  u32                  depth = 0;
  AABB2D               aabb{};
  bool                 leaf = true;
  static constexpr u32 N    = 64;
  union {
    Array<Pair<u32, AABB2D>> items;
    QuadNode *               children[4];
  };

  static QuadNode *create(AABB2D const &bound, u32 depth = 0) {
    QuadNode *out = new QuadNode;
    memset(out, 0, sizeof(QuadNode));
    out->aabb   = bound;
    out->leaf   = true;
    out->depth  = depth;
    g_max_depth = MAX(g_max_depth, depth);
    return out;
  }
  bool collides(AABB2D const &bound) { return aabb.collides(bound); }
  void remove_fast(u32 id, AABB2D const &bound) {
    if (!aabb.collides(bound)) return;
    if (leaf) {
      for (i32 i = 0; i < items.size; i++) {
        if (items[i].first == id) {
          items.remove_index_fast(i);
          i--;
        }
      }
    } else {
      ito(4) {
        jto(items.size) { children[i]->remove_fast(id, bound); }
      }
    }
  }
  void place(u32 id, AABB2D const &bound) {
    if (!aabb.collides(bound)) return;
    if (leaf) {
      if (items.size == N && depth < 12) {
        leaf      = false;
        float2 c  = aabb.center();
        float  dr = aabb.max_dim() / 4.0f;
        // swap because of the union
        Array<Pair<u32, AABB2D>> swap_items = items;
        ito(4) {
          float2 nc =
              c + dr * float2(float((i >> 0) & 1) * 2.0f - 1.0f, float((i >> 1) & 1) * 2.0f - 1.0f);
          AABB2D new_aabb;
          new_aabb.min = nc - float2(dr, dr);
          new_aabb.max = nc + float2(dr, dr);
          children[i]  = create(new_aabb, depth + 1);
          jto(swap_items.size) { children[i]->place(swap_items[j].first, swap_items[j].second); }
          children[i]->place(id, bound);
        }
        swap_items.release();
      } else {
        bool discard = false;
        for (i32 i = 0; i < items.size; i++) {
          if (items[i].first == id) {
            discard = true;
            break;
          }
        }
        if (!discard) items.push({id, bound});
      }
    } else {
      ito(4) children[i]->place(id, bound);
    }
  }
  template <typename T> void iter(float2 const &p, T f) {
    if (!aabb.inside(p)) return;
    if (leaf) {
      ito(items.size) f(items[i].first);
    } else {
      ito(4) children[i]->iter(p, f);
    }
  }
  void release() {
    if (leaf) items.release();
    if (!leaf) ito(4) children[i]->release();
    delete this;
  }
};
struct TriID {
  u32  v0, v1, v2;
  bool operator==(TriID const &that) const {
    return v0 == that.v0 && v1 == that.v1 && v2 == that.v2;
  }
};
static inline u64 hash_of(TriID id) {
  return ::hash_of(id.v0) ^ ::hash_of(id.v1) ^ ::hash_of(id.v2);
}

static thread_local RenderDoc_CTX *g_rctx = NULL;
static thread_local Array<float2>  g_failed_loop{};
static thread_local float2         g_failed_pnt{};
static thread_local double         g_triangulation_duration{};

template <typename T, typename K> static inline T bary(T A, T B, T C, K a, K b, K c) {
  return (A * a + B * b + C * c) / (a + b + c);
}
template <typename T, typename B>
static inline void calc_circumcenter(T v0, T v1, T v2, T &circumcenter, B &radius2) {
  // intersection of perpendicular bisectors
  B a          = dot2(v1 - v2);
  B b          = dot2(v2 - v0);
  B c          = dot2(v0 - v1);
  circumcenter = bary(v0, v1, v2, a * (b + c - a), b * (c + a - b), c * (a + b - c));
  T dr         = circumcenter - v0;
  radius2      = dot2(dr);
}

class Triangulation {
  public:
  private:
  class Bins {
private:
    Bins()  = default;
    ~Bins() = default;
    struct Bin {
  private:
      Bin()  = default;
      ~Bin() = default;

  public:
      SmallArray<u32, 128> items{};
      static Bin *         create() { return new Bin; }
      void                 release() {
        items.release();
        delete this;
      }
      template <typename T> void iter(T f) { ito(items.size) f(items[i]); }
    };
    Bin ** bins = NULL;
    u32    N    = 0;
    AABB2D aabb{};

public:
    static Bins *create(u32 N, AABB2D aabb) {
      if (N == 0) return NULL;
      Bins *out               = new Bins;
      out->N                  = N;
      out->bins               = new Bin *[N * N];
      out->aabb               = aabb;
      ito(N * N) out->bins[i] = NULL;
      return out;
    }

    void place(u32 id, AABB2D aabb) {
      if (this->aabb.collides(aabb) == false) return;
      float2 dr  = this->aabb.dim() / float(N);
      int2   min = int2((aabb.min - this->aabb.min) / dr);
      int2   max = int2((aabb.max - this->aabb.min) / dr);
      // int2   coord = glm::max(int2((aabb.min - this->aabb.min) / dr), int2(0, 0));
      // int2   dim   = glm::max(int2(ceil(aabb.dim() / dr)), int2(1, 1));
      i32 cnt = 0;
      for (i32 i = min.y; i <= max.y; i++) {
        for (i32 j = min.x; j <= max.x; j++) {
          i32 coordy = i;
          i32 coordx = j;
          if (coordx < 0 || coordy < 0) continue;
          if (coordx > N - 1 || coordy > N - 1) continue;
          u32 index = coordx + coordy * N;
          assert(index < N * N);
          if (bins[index] == NULL) bins[index] = Bin::create();
          bins[index]->items.push(id);
          cnt++;
        }
      }
      // if (!cnt) DebugBreak();
      assert(cnt);
    }

    template <typename T> void iter(float2 const &p, T f) {
      if (!this->aabb.inside(p)) return;
      float2 dr    = this->aabb.dim() / float(N);
      int2   coord = int2((p - this->aabb.min) / dr);
      if (coord.x < 0 || coord.y < 0) return;
      if (coord.x > N - 1 || coord.y > N - 1) return;
      u32 index = coord.x + coord.y * N;
      if (bins[index] == NULL) return;
      bins[index]->iter(f);
    }

    void release() {
      if (bins) {
        ito(N * N) if (bins[i]) bins[i]->release();
        delete[] bins;
      }
      delete this;
    }
  };

  DeltaTimer timer{};
  template <typename T> struct SlotArray {
    AutoArray<T, 0x1000> items{};
    // AutoArray<u32, 0x100> free_slots{};
    u32 add(T item) {
      /* if (free_slots.size) {
         u32 i          = free_slots.pop();
         items[i]       = item;
         items[i].alive = true;
         return i;
       }*/
      item.alive = true;
      items.push(item);
      return (u32)items.size - 1;
    }
    void remove(u32 i) {
      items[i].alive = false;
      // free_slots.push(i);
    }
  };
  struct HalfEdge {
    bool alive;
    u32  next;
    i32  neighbor;
    i32  f0, f1;
    u32  v0, v1;
  };
  struct Triangle {
    bool alive;
    // Marked for removal
    bool   bad;
    u32    v[3];
    u32    e[3];
    float2 circumcenter;
    float  radius2;
    AABB2D aabb;
    void   calc_circumcenter(float2 v0, float2 v1, float2 v2) {
      ::calc_circumcenter((float2)v0, (float2)v1, (float2)v2, circumcenter, radius2);

      // aabb.init(v0);
      // aabb.unite(v1);
      // aabb.unite(v2);

      float radius = sqrtf(radius2);
      aabb.min     = circumcenter - float2(radius, radius);
      aabb.max     = circumcenter + float2(radius, radius);
    }
    bool is_in_circle(float2 p) {
      float2 dr = circumcenter - (float2)p;
      return dot2(dr) < radius2 * (1.0f - 1.0e-4f);
    }
  };

  struct Vertex {
    float2              pos;
    SmallArray<u32, 16> edges{};
    void                release() { edges.release(); }
  };

  AutoArray<Vertex>   vertices{};
  SlotArray<Triangle> triangles{};
  SlotArray<HalfEdge> edges{};
  // <vertex_id, vertex_id> -> edge_id
  Hash_Table<Pair<u32, u32>, u32, Default_Allocator, 0x1000> edge_map{};

  // <vertex_id, vertex_id, vertex_id> -> triangle_id
  Hash_Table<TriID, u32, Default_Allocator, 0x1000> triangle_map{};
  // QuadNode *                                        quad_root = NULL;
  Bins *bins = NULL;
  // Is not called directly. only from remove_triangle
  void remove_edge(u32 id) {
    HalfEdge e = edges.items[id];
    if (e.neighbor >= 0) {
      HalfEdge &ne = edges.items[e.neighbor];
      assert(ne.alive && ne.neighbor == id);
      ne.neighbor = -1;
      // We remove both references to the edge and the face
      assert(ne.f1 == e.f0);
      ne.f1 = -1;
    }
    vertices[e.v0].edges.remove_fast(id);
    vertices[e.v1].edges.remove_fast(id);
    assert(edge_map.contains({e.v0, e.v1}));
    edge_map.remove({e.v0, e.v1});
    assert(!edge_map.contains({e.v0, e.v1}));
    edges.remove(id);
  }

  void remove_triangle(u32 tri_id) {

    Triangle &tri = triangles.items[tri_id];
    assert(tri.alive);
    // quad_root->remove_fast(tri_id, tri.aabb);
    assert(triangle_map.contains({tri.v[0], tri.v[1], tri.v[2]}));
    triangle_map.remove({tri.v[0], tri.v[1], tri.v[2]});
    // Remove edges and face references
    ito(3) remove_edge(tri.e[i]);
    triangles.remove(tri_id);
  }

  bool validate() {
    ito(triangles.items.size) {
      Triangle tri  = triangles.items[i];
      float2   v0   = vertices[tri.v[0]].pos;
      float2   v1   = vertices[tri.v[1]].pos;
      float2   v2   = vertices[tri.v[2]].pos;
      float2   e0   = v1 - v0;
      float2   e1   = v2 - v0;
      float    area = e0.x * e1.y - e0.y * e1.x;
      assert(area > 0.0f);
    }
    ito(edges.items.size) {
      HalfEdge e = edges.items[i];
      if (e.alive == false) continue;
      assert(edge_map.get({e.v0, e.v1}) == i);
      if (e.neighbor >= 0) {
        assert(e.f1 >= 0);
        HalfEdge ne = edges.items[e.neighbor];
        assert(ne.f1 == e.f0);
        assert(ne.f0 == e.f1);
        assert(ne.neighbor == i);
      }
      Triangle tri = triangles.items[e.f0];
      assert(tri.e[0] == i || tri.e[1] == i || tri.e[2] == i);
      assert(tri.e[0] == e.next || tri.e[1] == e.next || tri.e[2] == e.next);
    }
    ito(vertices.size) { assert(vertices[i].edges.size); }
    return true;
  }

  void add_triangle(u32 v0, u32 v1, u32 v2) {
    assert(!edge_map.contains({v0, v1}));
    assert(!edge_map.contains({v1, v2}));
    assert(!edge_map.contains({v2, v0}));
    assert(!triangle_map.contains(TriID{v0, v1, v2}));
    assert(([=]() -> bool {
      float2 p0   = vertices[v0].pos;
      float2 p1   = vertices[v1].pos;
      float2 p2   = vertices[v2].pos;
      float2 e0   = p1 - p0;
      float2 e1   = p2 - p0;
      float  area = e0.x * e1.y - e0.y * e1.x;
      // assert(area > 0.0f);
      return true;
    }()));

    u32       tri_id = triangles.add(Triangle{});
    Triangle &tri    = triangles.items[tri_id];
    u32       e0_id  = edges.add(HalfEdge{});
    u32       e1_id  = edges.add(HalfEdge{});
    u32       e2_id  = edges.add(HalfEdge{});

    HalfEdge &e0 = edges.items[e0_id];
    HalfEdge &e1 = edges.items[e1_id];
    HalfEdge &e2 = edges.items[e2_id];
    e0.v0        = v0;
    e0.v1        = v1;
    e1.v0        = v1;
    e1.v1        = v2;
    e2.v0        = v2;
    e2.v1        = v0;
    tri.e[0]     = e0_id;
    tri.e[1]     = e1_id;
    tri.e[2]     = e2_id;
    e0.f0        = tri_id;
    e1.f0        = tri_id;
    e2.f0        = tri_id;
    e0.next      = e1_id;
    e1.next      = e2_id;
    e2.next      = e0_id;
    tri.v[0]     = v0;
    tri.v[1]     = v1;
    tri.v[2]     = v2;
    tri.calc_circumcenter(vertices[v0].pos, vertices[v1].pos, vertices[v2].pos);
    // quad_root->place(tri_id, tri.aabb);
    bins->place(tri_id, tri.aabb);

    vertices[v0].edges.push(e0_id);
    vertices[v0].edges.push(e2_id);

    vertices[v1].edges.push(e0_id);
    vertices[v1].edges.push(e1_id);

    vertices[v2].edges.push(e1_id);
    vertices[v2].edges.push(e2_id);

    triangle_map.insert({v0, v1, v2}, tri_id);
    edge_map.insert({v0, v1}, e0_id);
    edge_map.insert({v1, v2}, e1_id);
    edge_map.insert({v2, v0}, e2_id);

    if (edge_map.contains({v1, v0})) {
      u32       ne0_id = edge_map.get({v1, v0});
      HalfEdge &ne0    = edges.items[ne0_id];
      ne0.neighbor     = e0_id;
      ne0.f1           = tri_id;
      e0.neighbor      = ne0_id;
      e0.f1            = ne0.f0;
    } else {
      e0.neighbor = -1;
      e0.f1       = -1;
    }

    if (edge_map.contains({v2, v1})) {
      u32       ne1_id = edge_map.get({v2, v1});
      HalfEdge &ne1    = edges.items[ne1_id];
      ne1.neighbor     = e1_id;
      ne1.f1           = tri_id;
      e1.neighbor      = ne1_id;
      e1.f1            = ne1.f0;
    } else {
      e1.neighbor = -1;
      e1.f1       = -1;
    }

    if (edge_map.contains({v0, v2})) {
      u32       ne2_id = edge_map.get({v0, v2});
      HalfEdge &ne2    = edges.items[ne2_id];
      ne2.neighbor     = e2_id;
      ne2.f1           = tri_id;
      e2.neighbor      = ne2_id;
      e2.f1            = ne2.f0;
    } else {
      e2.neighbor = -1;
      e2.f1       = -1;
    }
  }

  ~Triangulation() = default;

  // O(N)/O(1)
  void add_point(float2 p) {
    vertices.push({p, {}});
    u32 pnt_id = (u32)vertices.size - 1;

    // Triangles to remove
    SmallArray<u32, 16> bad_triangles{};
    bad_triangles.init();
    defer(bad_triangles.release());
    // We need the boundary to create new triangles to fill up the hole
    SmallArray<Pair<u32, u32>, 16> boundary_edges{};
    boundary_edges.init();
    defer(boundary_edges.release());

    // O(N) find all triangeles whose circumcircle contains the point
    // TODO(aschrein): optimize with a tree or smth.

    // ito(triangles.items.size) {
    //  if (triangles.items[i].alive == false) continue;
    //  if (triangles.items[i].is_in_circle(p)) {
    //    // mark for removal
    //    triangles.items[i].bad = true;
    //    bad_triangles.push(i);
    //  }
    //}

    // O(1) constant time retrival off a 2d bin array
    bins->iter(p, [&](u32 tri_id) {
      Triangle &tri = triangles.items[tri_id];
      if (tri.bad || !tri.alive) return;
      if (tri.is_in_circle(p)) {
        tri.bad = true;
        bad_triangles.push(tri_id);
      }
    });

    /* quad_root->iter(p, [&](u32 tri_id) {
       Triangle &tri = triangles.items[tri_id];
       if (tri.bad || tri.alive == false) return;
       if (tri.is_in_circle(p)) {
         tri.bad = true;
         bad_triangles.push(tri_id);
       }
     });*/
    assert(bad_triangles.size);
    if (bad_triangles.size == 0) return;
#if 0
        // The final list of triangles to be removed
    SmallArray<u32, 16> to_remove{};
    to_remove.init();
    defer(to_remove.release());
				i32 first_edge = -1;
    ito(bad_triangles.size) {
      Triangle &tri = triangles.items[bad_triangles[i]];
      jto(3) {
        HalfEdge &e0 = edges.items[tri.e[j]];
        // number of bad faces that touch this edge
        int bad_cnt = 0;
        bad_cnt += e0.f0 < 0 ? 0 : (triangles.items[e0.f0].bad) ? 1 : 0;
        bad_cnt += e0.f1 < 0 ? 0 : (triangles.items[e0.f1].bad) ? 1 : 0;
        // if one of the faces is not in the removal list then we have a boundary
        if (bad_cnt == 1) {
          first_edge = tri.e[j];
          goto get_loop;
          // we take the heighbor as this edge is going to be removed anyways
          // boundary_edges.push({e0.v0, e0.v1});
        }
      }
    }
  get_loop:
    assert(first_edge != -1);
    u32 cur_edge = first_edge;
    while (true) {
      HalfEdge &e0 = edges.items[cur_edge];
      to_remove.push(e0.f0);
      boundary_edges.push({e0.v0, e0.v1});
      Vertex &v         = vertices[e0.v1];
      u32     next_edge = cur_edge;
      ito(v.edges.size) {
        if (v.edges[i] == cur_edge) continue;
        HalfEdge &e0 = edges.items[v.edges[i]];
        // number of bad faces that touch this edge
        int bad_cnt = 0;
        bad_cnt += e0.f0 < 0 ? 0 : (triangles.items[e0.f0].bad) ? 1 : 0;
        bad_cnt += e0.f1 < 0 ? 0 : (triangles.items[e0.f1].bad) ? 1 : 0;
        if (bad_cnt == 1) {
          next_edge = v.edges[i];
          if (next_edge == first_edge) break;
        }
      }
      assert(next_edge != cur_edge);
      if (next_edge == first_edge) break;
      cur_edge = next_edge;
    }

    ito(to_remove.size) { remove_triangle(to_remove[i]); }
#else
    // Try to eliminate degenerate triangles from the iteration
    while (true) {
      bool discarded_any = false;
      for (i32 i = 0; i < bad_triangles.size; i++) {
        Triangle &tri     = triangles.items[bad_triangles[i]];
        bool      discard = false;
        jto(3) {
          HalfEdge &e0 = edges.items[tri.e[j]];
          // number of bad faces that touch this edge
          int bad_cnt = 0;
          bad_cnt += e0.f0 < 0 ? 0 : (triangles.items[e0.f0].bad) ? 1 : 0;
          bad_cnt += e0.f1 < 0 ? 0 : (triangles.items[e0.f1].bad) ? 1 : 0;
          // if one of the faces is not in the removal list then we have a boundary
          if (bad_cnt == 1) {
            float2 p0   = vertices[e0.v0].pos;
            float2 p1   = vertices[e0.v1].pos;
            float2 p2   = p;
            float2 e0   = p1 - p0;
            float2 e1   = p2 - p0;
            float  area = e0.x * e1.y - e0.y * e1.x;
            if (area < 0.0f) { // It's gonna be a degenerate triangle
              discard = true;
            }
          }
        }
        if (discard) {
          discarded_any                         = true;
          triangles.items[bad_triangles[i]].bad = false;
          bad_triangles.remove_index(i);
          i--;
        }
      }
      if (!discarded_any) break;
    }
    assert(bad_triangles.size);
    ito(bad_triangles.size) {
      Triangle &tri = triangles.items[bad_triangles[i]];
      jto(3) {
        HalfEdge &e0 = edges.items[tri.e[j]];
        // number of bad faces that touch this edge
        int bad_cnt = 0;
        bad_cnt += e0.f0 < 0 ? 0 : (triangles.items[e0.f0].bad) ? 1 : 0;
        bad_cnt += e0.f1 < 0 ? 0 : (triangles.items[e0.f1].bad) ? 1 : 0;
        // if one of the faces is not in the removal list then we have a boundary
        if (bad_cnt == 1) {
          boundary_edges.push({e0.v0, e0.v1});
        }
      }
    }

    ito(boundary_edges.size) {

      float2 p0   = vertices[boundary_edges[i].first].pos;
      float2 p1   = vertices[boundary_edges[i].second].pos;
      float2 p2   = vertices[pnt_id].pos;
      float2 e0   = p1 - p0;
      float2 e1   = p2 - p0;
      float  area = e0.x * e1.y - e0.y * e1.x;
      if (area < 0.0f) {
        float2 circumcenter;
        float  radius2;
        calc_circumcenter(p0, p1, p2, circumcenter, radius2);
        float radius = sqrt(radius2);
        int   N      = 1000;
        float dphi   = TWO_PI / (N - 1);
        jto(N) {
          g_failed_loop.push(circumcenter + radius * float2(cosf(dphi * j), sinf(dphi * j)));
          g_failed_loop.push(circumcenter +
                             radius * float2(cosf(dphi * (j + 1)), sinf(dphi * (j + 1))));
        }
        jto(boundary_edges.size) {
          g_failed_loop.push(vertices[boundary_edges[j].first].pos);
          g_failed_loop.push(vertices[boundary_edges[j].second].pos);
        }
        g_failed_pnt = p;
        return;
      }
    }

    ito(bad_triangles.size) { remove_triangle(bad_triangles[i]); }

#endif // 0

    // Now create new triangles
    ito(boundary_edges.size) {

      // if (edge_map.contains({boundary_edges[i].first, boundary_edges[i].second}) ||
      //    edge_map.contains({pnt_id, boundary_edges[i].first}) ||
      //    edge_map.contains({boundary_edges[i].second, pnt_id})) {
      //  jto(boundary_edges.size) {
      //    g_failed_loop.push(vertices[boundary_edges[j].first].pos);
      //    g_failed_loop.push(vertices[boundary_edges[j].second].pos);
      //  }
      //  g_failed_pnt = p;
      //  return;
      //}

      //{
      //  float2 p0   = vertices[boundary_edges[i].first].pos;
      //  float2 p1   = vertices[boundary_edges[i].second].pos;
      //  float2 p2   = vertices[pnt_id].pos;
      //  float2 e0   = p1 - p0;
      //  float2 e1   = p2 - p0;
      //  float  area = e0.x * e1.y - e0.y * e1.x;
      //  if (area < 0.0f) {
      //    jto(boundary_edges.size) {
      //      g_failed_loop.push(vertices[boundary_edges[j].first].pos);
      //      g_failed_loop.push(vertices[boundary_edges[j].second].pos);
      //    }
      //    g_failed_pnt = p;
      //    return;
      //  }
      //  // assert(area > 0.0f);
      //}

      add_triangle(boundary_edges[i].first, boundary_edges[i].second, pnt_id);
    }
  }

  void init(float2 *points, size_t num_points) {
    timer.init();
    timer.start_range();
    AABB2D aabb{};
    aabb.init(points[0]);
    ito(num_points) aabb.unite(points[i]);
    bins = Bins::create(32, {aabb.min - float2(1.0f, 1.0f), aabb.max + float2(1.0f, 1.0f)});
    // quad_root = QuadNode::create(aabb);
    triangles.items.reserve(num_points * 3);
    this->vertices.reserve(num_points + 4);
    edges.items.reserve(num_points * 3);
    float  r = aabb.max_dim() * 2.0f;
    float2 c = aabb.center();
    this->vertices.push({c + float2(-r, -r), {}});
    this->vertices.push({c + float2(r, -r), {}});
    this->vertices.push({c + float2(r, r), {}});
    this->vertices.push({c + float2(-r, r), {}});
    add_triangle(0, 1, 2);
    add_triangle(0, 2, 3);
    // float2 *sorted_points = new float2[num_points];
    // memcpy(sorted_points, points, sizeof(float2) * num_points);
    // defer(delete[] sorted_points);
    // quicky_sort(sorted_points, num_points,
    //            [](float2 const &a, float2 const &b) {
    //  assert(!any(isnan(a)) && !any(isnan(b)));
    //  return a.x < b.x; });
    // ito(num_points - 1) {
    //  assert(sorted_points[i].x < sorted_points[i + 1].x);
    //}

    ito(num_points) {
      add_point(points[i]);
      if (g_failed_loop.size) return;
    };
    timer.end_range();
    assert(validate());
    g_triangulation_duration += (timer.dt - g_triangulation_duration) * 0.5;
  }

  public:
  Array<TriID> getFaces() {
    Array<TriID> out{};
    ito(triangles.items.size) {
      if (triangles.items[i].alive) {
        if (triangles.items[i].v[0] < 4 || triangles.items[i].v[1] < 4 ||
            triangles.items[i].v[2] < 4)
          continue;
        out.push({triangles.items[i].v[0] - 4, triangles.items[i].v[1] - 4,
                  triangles.items[i].v[2] - 4});
      }
    }
    return out;
  }
  static Triangulation *create(float2 *points, size_t num_points) {
    Triangulation *out = new Triangulation;
    if (num_points == 0) return out;
    out->init(points, num_points);
    return out;
  }
  void release() {
    timer.release();
    edge_map.release();
    triangle_map.release();
    // quad_root->release();
    bins->release();
    delete this;
  }
};

class GBufferPass : public IPass {
  ~GBufferPass() = default;

  public:
  static constexpr char const *NAME = "GBuffer Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(normal_rt);                                                                             \
  RESOURCE(position_rt);                                                                           \
  RESOURCE(depth_rt);                                                                              \
  RESOURCE(gbuffer_vs);                                                                            \
  RESOURCE(gbuffer_ps);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  void release() override {
    timestamps.release(rctx->factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) rctx->factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
    delete this;
  }
#undef RESOURCE_LIST

  u32 width  = 0;
  u32 height = 0;
  // BufferThing bthing{};
  Random_Factory rf{685680485137855387, 15726070495360670683};
  // Random_Factory              rf{};
  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};
  RenderingContext *          rctx = NULL;

  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  static GBufferPass *create(RenderingContext *rctx) {
    GBufferPass *o = new GBufferPass;
    o->init(rctx);
    return o;
  }
  void init(RenderingContext *rctx) {
    this->rctx = rctx;
    auto dev   = rctx->factory;
    timestamps.init(dev);
    // bthing.init(dev);
    gbuffer_vs = dev->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 view_to_proj;
  float4x4 obj_to_view;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
  [[vk::location(3)]] float3 vpos    : TEXCOORD2;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] float3 normal  : NORMAL;
  [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.normal = mul(pc.obj_to_view, float4(input.normal.xyz, 0.0f)).xyz;
  output.uv     = input.uv;
  output.pos    = mul(pc.view_to_proj, mul(pc.obj_to_view, float4(input.pos, 1.0f)));
  output.vpos   = mul(pc.obj_to_view, float4(input.pos, 1.0f));
  return output;
}
)"),
                                    NULL, 0);
    gbuffer_ps = dev->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
  [[vk::location(3)]] float3 vpos    : TEXCOORD2;
};

struct PSOutput {
  float4 rt0 : SV_TARGET0;
  float4 rt1 : SV_TARGET1;
};

PSOutput main(in PSInput input) {
  PSOutput o;
  o.rt0 = float4(input.normal.xyz, 1.0f);
  o.rt1 = float4(input.vpos.xyz, 1.0f);
  return o;
}
)"),
                                    NULL, 0);
    signature  = [=] {
      rd::Binding_Space_Create_Info set_info{};
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      {
        rd::RT_Ref rt0{};
        rt0.format            = rd::Format::RGBA32_FLOAT;
        rt0.clear_color.clear = true;
        rt0.clear_color.r     = 0.0f;
        rt0.clear_color.g     = 0.0f;
        rt0.clear_color.b     = 0.0f;
        rt0.clear_color.a     = 0.0f;
        info.rts.push(rt0);
      }
      {
        rd::RT_Ref rt0{};
        rt0.format            = rd::Format::RGBA32_FLOAT;
        rt0.clear_color.clear = true;
        rt0.clear_color.r     = 0.0f;
        rt0.clear_color.g     = 0.0f;
        rt0.clear_color.b     = 0.0f;
        rt0.clear_color.a     = 0.0f;
        info.rts.push(rt0);
      }
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
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
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
      gfx_state.OM_set_blend_state(1, bs);
      gfx_state.VS_set_shader(gbuffer_vs);
      gfx_state.PS_set_shader(gbuffer_ps);
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
      {
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
      }
      gfx_state.IA_set_vertex_binding(0, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();
  }
  void update_frame_buffer(RenderingContext *rctx) {
    auto dev = rctx->factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (normal_rt.is_valid()) dev->release_resource(normal_rt);
    if (depth_rt.is_valid()) dev->release_resource(depth_rt);
    if (position_rt.is_valid()) dev->release_resource(position_rt);
    position_rt = [=] {
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
    normal_rt = [=] {
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
      rt0.image  = normal_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);
      rt0.image  = position_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);
      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render() {
    auto dev = rctx->factory;
    timestamps.update(dev);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f,
    // 0.0f));
    // bthing.test_buffers(dev);
    u32 width  = rctx->config->get_u32("g_buffer_width");
    u32 height = rctx->config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    struct PushConstants {
      float4x4 view_to_proj;
      float4x4 obj_to_view;
    } pc;

    float4x4 view_to_proj  = rctx->gizmo_layer->get_camera().proj;
    float4x4 world_to_view = rctx->gizmo_layer->get_camera().view;

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "GBuffer Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      pc.view_to_proj = view_to_proj;

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      ctx->bind_table(table);
      ctx->bind_graphics_pso(pso);

      if (rctx->config->get_bool("gizmo.render_test")) {
        static Array<float2>  points{};
        static Triangulation *triangulation = NULL;
        static PCG            state{};
        if (g_failed_loop.size) {

        } else {
          if (rctx->config->get_bool("gizmo.render_test.update")) {
            if (triangulation) triangulation->release();
            points.release();
            points.reserve(1000);
            state = rf.get_state();
            ito(1000) { points.push(rf.rand_sphere_center().xy); }
            triangulation = Triangulation::create(&points[0], points.size);
          }
        }
        if (triangulation) {
          Array<TriID> tris = triangulation->getFaces();
          if (rctx->config->get_bool("gizmo.render_test.src")) {
            ito(tris.size) {
              TriID tri = tris[i];
              rctx->gizmo_layer->draw_line(float3(points[tri.v0], 0.0f).xzy,
                                           float3(points[tri.v1], 0.0f).xzy,
                                           float3(1.0f, 1.0f, 0.0f));
              rctx->gizmo_layer->draw_line(float3(points[tri.v1], 0.0f).xzy,
                                           float3(points[tri.v2], 0.0f).xzy,
                                           float3(1.0f, 1.0f, 0.0f));
              rctx->gizmo_layer->draw_line(float3(points[tri.v2], 0.0f).xzy,
                                           float3(points[tri.v0], 0.0f).xzy,
                                           float3(1.0f, 1.0f, 0.0f));
            }
          }
          tris.release();
          if (rctx->config->get_bool("gizmo.render_test.err")) {
            if (g_failed_loop.size) {
              rctx->gizmo_layer->draw_sphere(float3(g_failed_pnt, 0.0f).xzy, 0.002f,
                                             float3(0.0f, 1.0f, 1.0f));
              ito(g_failed_loop.size / 2) {
                float2 p0 = g_failed_loop[i * 2 + 0];
                float2 p1 = g_failed_loop[i * 2 + 1];
                rctx->gizmo_layer->draw_line(float3(p0, 0.0f).xzy, float3(p1, 0.0f).xzy,
                                             float3(1.0f, 0.0f, 0.0f));
              }
            }
          }
        }
      }

      if (rctx->config->get_bool("ras.render_meshlets")) {
        rctx->scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            if (auto *sc = mn->getComponent<GfxMeshletSufraceComponent>()) {
              ito(mn->getNumSurfaces()) {
                GfxMeshletSurface *gfx_meshlets = sc->get_meshlets(i);
                gfx_meshlets->iterate([](Meshlet const &meshlet) {

                });
              }
            }
          }
        });
      }
      if (rctx->config->get_bool("ras.render_geometry")) {
        rctx->scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            GfxSufraceComponent *gs           = mn->getComponent<GfxSufraceComponent>();
            float4x4             obj_to_world = mn->get_transform();
            pc.obj_to_view                    = world_to_view * obj_to_world;
            table->push_constants(&pc, 0, sizeof(pc));
            ito(gs->getNumSurfaces()) {
              GfxSurface *s = gs->getSurface(i);
              s->draw(ctx, gfx_state);
            }
          }
        });
      }

      if (rctx->config->get_bool("gizmo.enable")) {
        auto g_camera = rctx->gizmo_layer->get_camera();
        {
          float dx = 1.0e-1f * g_camera.distance;
          rctx->gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
          rctx->gizmo_layer->draw_cylinder(g_camera.look_at,
                                           g_camera.look_at + float3{dx, 0.0f, 0.0f}, dx * 0.04f,
                                           float3{1.0f, 0.0f, 0.0f});
          rctx->gizmo_layer->draw_cylinder(g_camera.look_at,
                                           g_camera.look_at + float3{0.0f, dx, 0.0f}, dx * 0.04f,
                                           float3{0.0f, 1.0f, 0.0f});
          rctx->gizmo_layer->draw_cylinder(g_camera.look_at,
                                           g_camera.look_at + float3{0.0f, 0.0f, dx}, dx * 0.04f,
                                           float3{0.0f, 0.0f, 1.0f});
        }

        if (rctx->config->get_bool("gizmo.render_bounds")) {
          rctx->scene->traverse([&](Node *node) {
            AABB     aabb = node->getAABB();
            float4x4 t(1.0f);
            rctx->gizmo_layer->render_linebox(transform(t, aabb.min), transform(t, aabb.max),
                                              float3(1.0f, 0.0f, 0.0f));
          });
        }
        if (rctx->config->get_bool("gizmo.render_bvh")) {
          rctx->scene->traverse([&](Node *node) {
            if (MeshNode *mn = node->dyn_cast<MeshNode>()) {

              if (auto *sc = mn->getComponent<BVHSufraceComponent>()) {
                if (sc->getBVH()) {
                  render_bvh(float4x4(1.0f), sc->getBVH(), rctx->gizmo_layer);
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

      float4x4 viewproj = rctx->gizmo_layer->get_camera().viewproj();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      rctx->gizmo_layer->render(ctx, width, height);

      rctx->gizmo_layer->reset();
      timestamps.end_range(ctx);
    }
    ctx->end_render_pass();
    Resource_ID e = dev->end_render_pass(ctx);
    timestamps.commit(e);
  }
  GBuffer get_gbuffer() {
    GBuffer out{};
    out.normal = normal_rt;
    out.depth  = depth_rt;
    return out;
  }

  char const *getName() override { return NAME; }
  u32         getNumBuffers() override { return 3; }
  char const *getBufferName(u32 i) override {
    if (i == 0) return "GBuffer.Normal";
    if (i == 1) return "GBuffer.VPos";
    return "GBuffer.Depth";
  }
  Resource_ID getBuffer(u32 i) override {
    if (i == 0) return normal_rt;
    if (i == 1) return position_rt;
    return depth_rt;
  }
  double getLastDurationInMs() override { return timestamps.duration; }
};

#if 1

class Event_Consumer : public IGUIApp, public IPassMng {
  public:
  Hash_Table<u64, float4>     minmax{};
  InlineArray<IPass *, 0x100> passes{};
  // GBufferPass gbuffer_pass;
  // GizmoPass   gizmo_pass;
  // ComposePass compose_pass;
  IPass *getPass(char const *name) override {
    ito(passes.size) if (strcmp(passes[i]->getName(), name) == 0) return passes[i];
    return NULL;
  }
  RenderingContext *rctx = NULL;
  void              init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("camera")) {
        rctx->gizmo_layer->get_camera().traverse(l->next);
      } else if (l->cmp_symbol("config")) {
        rctx->config->traverse(l->next);
      } else if (l->cmp_symbol("scene")) {
        rctx->scene->restore(l);
      }
    }
  }
  void on_gui() override { //
    // timer.();
    ImGui::Begin("Scene");
    {
      String_Builder sb;
      sb.init();
      defer(sb.release());
      rctx->scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        rctx->scene->restore(cur);
      }
    }
    ImGui::End();

    ImGui::Begin("Config");
    if (rctx->config->on_imgui()) rctx->dump();
    ito(passes.size) {
      ImGui::Text("%s %fms", passes[i]->getName(), passes[i]->getLastDurationInMs());
    }
    ImGui::Text("g_triangulation_duration %fms", g_triangulation_duration);

    if (ImGui::Button("Rebuild BVH")) {
      rctx->scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (mn->getComponent<BVHSufraceComponent>()) {
            mn->getComponent<BVHSufraceComponent>()->updateBVH();
          }
        }
      });
    }
    ImGui::End();

    ito(passes.size) {
      jto(passes[i]->getNumBuffers()) {
        ImGui::Begin(passes[i]->getBufferName(j));
        {
          rctx->gizmo_layer->per_imgui_window();
          auto    wsize = get_window_size();
          float4 *m     = minmax.get_or_null(passes[i]->getBuffer(j).data);
          if (m == NULL)
            minmax.insert(passes[i]->getBuffer(j).data, float4{0.0f, 1.0f, 0.0f, 0.0f});
          m = minmax.get_or_null(passes[i]->getBuffer(j).data);
          ImGui::SetNextItemWidth(100);
          ImGui::DragFloat("min", &m->x, 1.0e-3f);
          ImGui::SameLine();
          ImGui::SetNextItemWidth(100);
          ImGui::DragFloat("max", &m->y, 1.0e-3f);
          ImGui::SameLine();
          ImGui::SetNextItemWidth(100);
          int mip = m->z;
          ImGui::DragInt("mip", &mip, 1);
          m->z = float(mip);
          ImGui::Image(
              bind_texture(passes[i]->getBuffer(j), 0, mip, rd::Format::NATIVE, m->x, m->y),
              ImVec2(wsize.x, wsize.y - 20.0f));
          { Ray ray = rctx->gizmo_layer->getMouseRay(); }
        }
        ImGui::End();
      }
      ImGui::Text("%s %fms", passes[i]->getName(), passes[i]->getLastDurationInMs());
    }
  }
  void on_init() override { //
    minmax.init();
    rctx          = new RenderingContext;
    rctx->factory = this->factory;
    TMP_STORAGE_SCOPE;

    // new XYZDragGizmo(gizmo_layer, &pos);
    rctx->scene  = Scene::create();
    rctx->config = new Config;
    rctx->config->init(stref_s(R"(
 (
  (add u32  g_buffer_width 512 (min 4) (max 2048))
  (add u32  g_buffer_height 512 (min 4) (max 2048))
  (add u32  baking.size 512 (min 4) (max 4096))
  (add bool G.I.color_triangles 0)
 )
 )"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/ssr_test.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/norradalur-froyar/scene.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/cut.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/untitled.gltf"));
    // rctx->scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
    rctx->scene->update();
    rctx->scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        GfxSufraceComponent::create(rctx->factory, mn);
        // MeshletSufraceComponent::create(mn, 255, 256);
        // GfxMeshletSufraceComponent::create(factory, mn);
      }
    });
    rctx->pass_mng = this;
    passes.push(GBufferPass::create(rctx));

    rctx->gizmo_layer =
        Gizmo_Layer::create(factory, ((GBufferPass *)getPass(GBufferPass::NAME))->pass);
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
  }
  void on_release() override { //
    minmax.release();
    rctx->dump();
    rctx->gizmo_layer->release();
    rctx->scene->release();
    rctx->config->release();
    ito(passes.size) passes[i]->release();
    passes.release();
    delete rctx->config;
    delete rctx;
  }
  void on_frame() override { //
    rctx->scene->get_root()->update();
    ito(passes.size) passes[i]->render();
  }
};
#endif
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto window_loop = [](rd::Impl_t impl) { IGUIApp::start<Event_Consumer>(impl); };
  // std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  // std::thread dx12_thread = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  // vulkan_thread.join();
  // dx12_thread.join();

  // window_loop(rd::Impl_t::VULKAN);
  window_loop(rd::Impl_t::DX12);
  return 0;
}