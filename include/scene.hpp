#ifndef SCENE_HPP
#define SCENE_HPP

#include "rendering.hpp"
#include "utils.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace glm;

using int2     = ivec2;
using int3     = ivec3;
using int4     = ivec4;
using uint2    = uvec2;
using uint3    = uvec3;
using uint4    = uvec4;
using float2   = vec2;
using float3   = vec3;
using float4   = vec4;
using float2x2 = mat2;
using float3x3 = mat3;
using float4x4 = mat4;

#ifdef __clang__
#define ALIGN16 __attribute__((aligned(16)))
#else
#define ALIGN16 __declspec(align(16))
#endif

typedef ivec2 ALIGN16 aint2;
typedef ivec3 ALIGN16 aint3;
typedef ivec4 ALIGN16 aint4;
typedef uvec2 ALIGN16 auint2;
typedef uvec3 ALIGN16 auint3;
typedef uvec4 ALIGN16 auint4;
typedef vec2 ALIGN16  afloat2;
typedef vec3 ALIGN16  afloat3;
typedef vec4 ALIGN16  afloat4;
typedef mat2 ALIGN16  afloat2x2;
typedef mat3 ALIGN16  afloat3x3;
typedef mat4 ALIGN16  afloat4x4;

static constexpr float PI                  = 3.1415926;
static constexpr float TWO_PI              = 6.2831852;
static constexpr float FOUR_PI             = 12.566370;
static constexpr float INV_PI              = 0.3183099;
static constexpr float INV_TWO_PI          = 0.1591549;
static constexpr float INV_FOUR_PI         = 0.0795775;
static constexpr float DIELECTRIC_SPECULAR = 0.04;

// https://github.com/graphitemaster/normals_revisited
static float minor(const float m[16], int r0, int r1, int r2, int c0, int c1,
                   int c2) {
  return m[4 * r0 + c0] * (m[4 * r1 + c1] * m[4 * r2 + c2] -
                           m[4 * r2 + c1] * m[4 * r1 + c2]) -
         m[4 * r0 + c1] * (m[4 * r1 + c0] * m[4 * r2 + c2] -
                           m[4 * r2 + c0] * m[4 * r1 + c2]) +
         m[4 * r0 + c2] * (m[4 * r1 + c0] * m[4 * r2 + c1] -
                           m[4 * r2 + c0] * m[4 * r1 + c1]);
}

static void cofactor(const float src[16], float dst[16]) {
  dst[0]  = minor(src, 1, 2, 3, 1, 2, 3);
  dst[1]  = -minor(src, 1, 2, 3, 0, 2, 3);
  dst[2]  = minor(src, 1, 2, 3, 0, 1, 3);
  dst[3]  = -minor(src, 1, 2, 3, 0, 1, 2);
  dst[4]  = -minor(src, 0, 2, 3, 1, 2, 3);
  dst[5]  = minor(src, 0, 2, 3, 0, 2, 3);
  dst[6]  = -minor(src, 0, 2, 3, 0, 1, 3);
  dst[7]  = minor(src, 0, 2, 3, 0, 1, 2);
  dst[8]  = minor(src, 0, 1, 3, 1, 2, 3);
  dst[9]  = -minor(src, 0, 1, 3, 0, 2, 3);
  dst[10] = minor(src, 0, 1, 3, 0, 1, 3);
  dst[11] = -minor(src, 0, 1, 3, 0, 1, 2);
  dst[12] = -minor(src, 0, 1, 2, 1, 2, 3);
  dst[13] = minor(src, 0, 1, 2, 0, 2, 3);
  dst[14] = -minor(src, 0, 1, 2, 0, 1, 3);
  dst[15] = minor(src, 0, 1, 2, 0, 1, 2);
}

static float4x4 cofactor(float4x4 const &in) {
  float4x4 out;
  cofactor(&in[0][0], &out[0][0]);
  return out;
}

static inline float halton(int i, int base) {
  float x = 1.0f / base, v = 0.0f;
  while (i > 0) {
    v += x * (i % base);
    i = floor(i / base);
    x /= base;
  }
  return v;
}

struct PCG {
  u64 state = 0x853c49e6748fea9bULL;
  u64 inc   = 0xda3e39cb94b95bdbULL;
  u32 next() {
    uint64_t oldstate   = state;
    state               = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    int      rot        = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }
  f64 nextf() { return double(next()) / UINT32_MAX; }
};

class Random_Factory {
  public:
  float  rand_unit_float() { return float(pcg.nextf()); }
  float3 rand_unit_cube() {
    return float3{rand_unit_float() * 2.0 - 1.0, rand_unit_float() * 2.0 - 1.0,
                  rand_unit_float() * 2.0 - 1.0};
  }
  // Random unsigned integer in the range [begin, end)
  u32 uniform(u32 begin, u32 end) {
    ASSERT_PANIC(end > begin);
    u32 range = end - begin;
    u32 mod   = UINT32_MAX % range;
    if (mod == 0) return (pcg.next() % range) + begin;
    // Kill the bias
    u32 new_max = UINT32_MAX - mod;
    while (true) {
      u32 rand = pcg.next();
      if (rand > new_max) continue;
      return (rand % range) + begin;
    }
  }
  // Z is up here
  float3 polar_to_cartesian(float sinTheta, float cosTheta, float sinPhi,
                            float cosPhi) {
    return float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
  }
  // Z is up here
  float3 uniform_sample_cone(float cos_theta_max, float3 xbasis, float3 ybasis,
                             float3 zbasis) {
    vec2   rand     = vec2(rand_unit_float(), rand_unit_float());
    float  cosTheta = (1.0f - rand.x) + rand.x * cos_theta_max;
    float  sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    float  phi      = rand.y * PI * 2.0f;
    float3 samplev = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
  }

  float3 rand_unit_sphere() {
    while (true) {
      float3 pos = rand_unit_cube();
      if (dot(pos, pos) <= 1.0f) return pos;
    }
  }

  float3 rand_unit_sphere_surface() {
    while (true) {
      float3 pos  = rand_unit_cube();
      f32    len2 = dot(pos, pos);
      if (len2 <= 1.0f) return pos / std::sqrt(len2);
    }
  }

  float3 sample_lambert_BRDF(float3 N) {
    return normalize(N + rand_unit_sphere());
  }

  vec2 random_halton() {
    f32 u = halton(halton_id + 1, 2);
    f32 v = halton(halton_id + 1, 3);
    halton_id++;
    return vec2(u, v);
  }

  static float3 SampleHemisphere_Cosinus(float2 xi) {
    float phi      = xi.y * 2.0 * PI;
    float cosTheta = std::sqrt(1.0 - xi.x);
    float sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

    return vec3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
  }

  private:
  PCG pcg;
  u32 halton_id = 0;
};

struct Image2D_Raw {
  u32        width;
  u32        height;
  rd::Format format;
  u8 *       data;
  void       init(u32 width, u32 height, rd::Format format, u8 *data) {
    MEMZERO(*this);
    this->width  = width;
    this->height = height;
    this->format = format;
    u32 size     = get_bpp() * width * height;
    this->data   = (u8 *)tl_alloc(size);
    memcpy(this->data, data, size);
  }
  void release() {
    if (data != NULL) tl_free(data);
    MEMZERO(*this);
  }
  u32 get_bpp() {
    switch (format) {
    case rd::Format::RGBA8_UNORM:
    case rd::Format::RGBA8_SRGBA: return 4u;
    case rd::Format::RGB32_FLOAT: return 12u;
    default: ASSERT_PANIC(false && "unsupported format");
    }
  }
  vec4 load(uint2 coord) {
    u32 bpc = 4u;
    switch (format) {
    case rd::Format::RGBA8_UNORM:
    case rd::Format::RGBA8_SRGBA: bpc = 4u; break;
    case rd::Format::RGB32_FLOAT: bpc = 12u; break;
    default: ASSERT_PANIC(false && "unsupported format");
    }
    auto load_f32 = [&](uint2 coord, u32 component) {
      uint2 size = uint2(width, height);
      return *(
          f32 *)&data[coord.x * bpc + coord.y * size.x * bpc + component * 4u];
    };
    uint2 size = uint2(width, height);
    if (coord.x >= size.x) coord.x = size.x - 1;
    if (coord.y >= size.y) coord.y = size.y - 1;
    switch (format) {
    case rd::Format::RGBA8_UNORM: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];
      return vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                  float(a) / 255.0f);
    }
    case rd::Format::RGBA8_SRGBA: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];

      auto out = vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                      float(a) / 255.0f);
      out.r    = std::pow(out.r, 2.2f);
      out.g    = std::pow(out.g, 2.2f);
      out.b    = std::pow(out.b, 2.2f);
      out.a    = std::pow(out.a, 2.2f);
      return out;
    }
    case rd::Format::RGB32_FLOAT: {
      f32 r = load_f32(coord, 0u);
      f32 g = load_f32(coord, 1u);
      f32 b = load_f32(coord, 2u);
      return vec4(r, g, b, 1.0f);
    }
    default: ASSERT_PANIC(false && "unsupported format");
    }
  };
  vec4 sample(vec2 uv) {
    ivec2 size    = ivec2(width, height);
    vec2  suv     = uv * vec2(float(size.x - 1u), float(size.y - 1u));
    ivec2 coord[] = {
        ivec2(i32(suv.x), i32(suv.y)),
        ivec2(i32(suv.x), i32(suv.y + 1.0f)),
        ivec2(i32(suv.x + 1.0f), i32(suv.y)),
        ivec2(i32(suv.x + 1.0f), i32(suv.y + 1.0f)),
    };
    ito(4) {
      // Repeat
      jto(2) {
        while (coord[i][j] >= size[j]) coord[i][j] -= size[j];
        while (coord[i][j] < 0) coord[i][j] += size[j];
      }
    }
    vec2  fract = vec2(suv.x - std::floor(suv.x), suv.y - std::floor(suv.y));
    float weights[] = {
        (1.0f - fract.x) * (1.0f - fract.y),
        (1.0f - fract.x) * (fract.y),
        (fract.x) * (1.0f - fract.y),
        (fract.x) * (fract.y),
    };
    vec4 result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    ito(4) result += load(uint2(coord[i].x, coord[i].y)) * weights[i];
    return result;
  };
};

static inline float3 safe_normalize(float3 v) {
  return v / (glm::length(v) + 1.0e-5f);
}

struct Vertex_Full {
  float3      position;
  float3      normal;
  float3      binormal;
  float3      tangent;
  float2      u0;
  float2      u1;
  float2      u2;
  float2      u3;
  Vertex_Full transform(float4x4 const &transform) {
    Vertex_Full out;
    float4x4    cmat = cofactor(transform);
    out.position     = float3(transform * float4(position, 1.0f));
    out.normal       = safe_normalize(float3(cmat * float4(normal, 0.0f)));
    out.tangent      = safe_normalize(float3(cmat * float4(tangent, 0.0f)));
    out.binormal     = safe_normalize(float3(cmat * float4(binormal, 0.0f)));
    out.u0           = u0;
    out.u1           = u1;
    out.u2           = u2;
    out.u3           = u3;
    return out;
  }
};
struct Tri_Index {
  u32 i0, i1, i2;
};
struct Triangle_Full {
  Vertex_Full v0;
  Vertex_Full v1;
  Vertex_Full v2;
};
// We are gonna use one simplified material schema for everything
struct PBR_Material {
  // AO+Roughness+Metalness
  i32    normal_id;
  i32    albedo_id;
  i32    arm_id;
  f32    metal_factor;
  f32    roughness_factor;
  float4 albedo_factor;
  void   init() {
    normal_id        = -1;
    albedo_id        = -1;
    arm_id           = -1;
    metal_factor     = 1.0f;
    roughness_factor = 1.0f;
    albedo_factor    = float4(1.0f);
  }
};

struct Attribute {
  rd::Attriute_t type;
  rd::Format     format;
  u32            offset;
  u32            stride;
};

struct Raw_Mesh_Opaque {
  Array<u8>                  attribute_data;
  Array<u8>                  index_data;
  rd::Index_t                index_type;
  InlineArray<Attribute, 16> attributes;
  u32                        num_vertices;
  u32                        num_indices;
  float3                     min;
  float3                     max;

  void init() {
    MEMZERO(*this);
    attributes.init();
    attribute_data.init();
    index_data.init();
  }
  void release() {
    attributes.release();
    attribute_data.release();
    index_data.release();
  }

  Attribute get_attribute(rd::Attriute_t type) {
    ito(attributes.size) {
      if (attributes[i].type == type) return attributes[i];
    }
    TRAP;
  }

  float3 fetch_position(u32 index) {
    ito(attributes.size) {
      switch (attributes[i].type) {

      case rd::Attriute_t ::POSITION:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        float3 pos;
        memcpy(&pos,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               12);
        return pos;
      default: break;
      }
    }
    TRAP;
  }
  Vertex_Full fetch_vertex(u32 index) {
    Vertex_Full v;
    MEMZERO(v);
    ito(attributes.size) {
      switch (attributes[i].type) {
      case rd::Attriute_t ::NORMAL:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.normal,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::BINORMAL:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.binormal,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::TANGENT:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.tangent,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::POSITION:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.position,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::TEXCOORD0:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u0,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               8);
        break;
      case rd::Attriute_t ::TEXCOORD1:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u1,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               8);
        break;
      case rd::Attriute_t ::TEXCOORD2:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u2,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               8);
        break;
      case rd::Attriute_t ::TEXCOORD3:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u3,
               attribute_data.at(index * attributes[i].stride +
                                 attributes[i].offset),
               8);
        break;
      default: TRAP;
      }
    }
    return v;
  }

  Tri_Index get_tri_index(u32 id) {
    Tri_Index o;
    if (index_type == rd::Index_t::UINT16) {
      o.i0 = (u32) * (u16 *)index_data.at(2 * (id * 3 + 0));
      o.i1 = (u32) * (u16 *)index_data.at(2 * (id * 3 + 1));
      o.i2 = (u32) * (u16 *)index_data.at(2 * (id * 3 + 2));
    } else {
      o.i0 = (u32) * (u32 *)index_data.at(4 * (id * 3 + 0));
      o.i1 = (u32) * (u32 *)index_data.at(4 * (id * 3 + 1));
      o.i2 = (u32) * (u32 *)index_data.at(4 * (id * 3 + 2));
    }
    return o;
  }

  Triangle_Full fetch_triangle(u32 id) {
    Tri_Index   tind = get_tri_index(id);
    Vertex_Full v0   = fetch_vertex(tind.i0);
    Vertex_Full v1   = fetch_vertex(tind.i1);
    Vertex_Full v2   = fetch_vertex(tind.i2);
    return {v0, v1, v2};
  }
  Vertex_Full interpolate_vertex(u32 index, float2 uv) {
    Triangle_Full face = fetch_triangle(index);
    Vertex_Full   v0   = face.v0;
    Vertex_Full   v1   = face.v1;
    Vertex_Full   v2   = face.v2;
    float         k1   = uv.x;
    float         k2   = uv.y;
    float         k0   = 1.0f - uv.x - uv.y;
    Vertex_Full   vertex;
    vertex.normal =
        safe_normalize(v0.normal * k0 + v1.normal * k1 + v2.normal * k2);
    vertex.position = v0.position * k0 + v1.position * k1 + v2.position * k2;
    vertex.tangent =
        safe_normalize(v0.tangent * k0 + v1.tangent * k1 + v2.tangent * k2);
    vertex.binormal =
        safe_normalize(v0.binormal * k0 + v1.binormal * k1 + v2.binormal * k2);
    vertex.u0 = v0.u0 * k0 + v1.u0 * k1 + v2.u0 * k2;
    vertex.u1 = v0.u1 * k0 + v1.u1 * k1 + v2.u1 * k2;
    vertex.u2 = v0.u2 * k0 + v1.u2 * k1 + v2.u2 * k2;
    vertex.u3 = v0.u3 * k0 + v1.u3 * k1 + v2.u3 * k2;
    return vertex;
  }
};

struct Transform_Node {
  float3     offset;
  quat       rotation;
  float3     scale;
  float4x4   transform_cache;
  Array<u32> meshes;
  Array<u32> children;
  void       init() {
    MEMZERO(*this);
    meshes.init();
    children.init();
    scale           = float3(1.0f, 1.0f, 1.0f);
    offset          = float3(0.0f, 0.0f, 0.0f);
    transform_cache = float4x4(1.0f);
  }
  void release() {
    meshes.release();
    children.release();
    MEMZERO(*this);
  }
  void update_cache(float4x4 const &parent = float4x4(1.0f)) {
    transform_cache = parent * get_transform();
  }
  float4x4 get_transform() {
    //  return transform;
    return glm::translate(float4x4(1.0f), offset) * (float4x4)rotation *
           glm::scale(float4x4(1.0f), scale);
  }
  float4x4 get_cofactor() {
    mat4 out{};
    mat4 transform = get_transform();
    cofactor(&transform[0][0], &out[0][0]);
  }
};

// To make things simple we use one format of meshes
struct PBR_Model {
  Array<Image2D_Raw>     images;
  Array<Raw_Mesh_Opaque> meshes;
  Array<PBR_Material>    materials;
  Array<Transform_Node>  nodes;

  void init() {
    images.init();
    meshes.init();
    materials.init();
    nodes.init();
  }
  void release() {
    ito(images.size) images[i].release();
    images.release();
    ito(meshes.size) meshes[i].release();
    meshes.release();
    materials.release();
    ito(nodes.size) nodes[i].release();
    nodes.release();
  }
};

PBR_Model   load_gltf_pbr(string_ref filename);
Image2D_Raw load_image(string_ref filename,
                       rd::Format format = rd::Format::RGBA8_SRGBA);
#endif // SCENE