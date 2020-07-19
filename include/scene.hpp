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

typedef i32 ALIGN16   ai32;
typedef u32 ALIGN16   au32;
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
    if (data != NULL) memcpy(this->data, data, size);
  }
  u32  get_size_in_bytes() const { return get_bpp() * width * height; }
  void release() {
    if (data != NULL) tl_free(data);
    MEMZERO(*this);
  }
  u32 get_bpp() const {
    switch (format) {
    case rd::Format::RGBA8_UNORM:
    case rd::Format::RGBA8_SRGBA: return 4u;
    case rd::Format::RGB32_FLOAT: return 12u;
    default: ASSERT_PANIC(false && "unsupported format");
    }
  }
  vec4 load(int2 coord) const {
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
    if (coord.x < 0) coord.x = 0;
    if (coord.y < 0) coord.y = 0;
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
  }
  void write(int2 coord, float4 pixel) {
    u32 bpc = 4u;
    switch (format) {
    case rd::Format::RGBA8_UNORM:
    case rd::Format::RGBA8_SRGBA: bpc = 4u; break;
    case rd::Format::RGB32_FLOAT: bpc = 12u; break;
    case rd::Format::RGBA32_FLOAT: bpc = 16u; break;
    default: ASSERT_PANIC(false && "unsupported format");
    }
    uint2 size = uint2(width, height);
    if (coord.x >= size.x) coord.x = size.x - 1;
    if (coord.x < 0) coord.x = 0;
    if (coord.y < 0) coord.y = 0;
    if (coord.y >= size.y) coord.y = size.y - 1;
    switch (format) {
      // clang-format off
    case rd::Format::RGBA8_UNORM: {
      data[coord.x * bpc + coord.y * size.x * bpc + 0u] = (u8)((float)clamp(pixel.r, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 1u] = (u8)((float)clamp(pixel.g, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 2u] = (u8)((float)clamp(pixel.b, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 3u] = (u8)((float)clamp(pixel.a, 0.0f, 1.0f) * 255.0f);
      break;
    }
    case rd::Format::RGBA8_SRGBA: {
      pixel.r    = std::pow(pixel.r, 1.0f/2.2f);
      pixel.g    = std::pow(pixel.g, 1.0f/2.2f);
      pixel.b    = std::pow(pixel.b, 1.0f/2.2f);
      pixel.a    = std::pow(pixel.a, 1.0f/2.2f);
      data[coord.x * bpc + coord.y * size.x * bpc + 0u] = (u8)((float)clamp(pixel.r, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 1u] = (u8)((float)clamp(pixel.g, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 2u] = (u8)((float)clamp(pixel.b, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 3u] = (u8)((float)clamp(pixel.a, 0.0f, 1.0f) * 255.0f);
      break;
    }
    case rd::Format::RGB32_FLOAT: {
      data[coord.x * bpc + coord.y * size.x * bpc + 0u] = (u8)((float)clamp(pixel.r, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 1u] = (u8)((float)clamp(pixel.g, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 2u] = (u8)((float)clamp(pixel.b, 0.0f, 1.0f) * 255.0f);
      break;
    }
    case rd::Format::RGBA32_FLOAT: {
      data[coord.x * bpc + coord.y * size.x * bpc + 0u] = (u8)((float)clamp(pixel.r, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 1u] = (u8)((float)clamp(pixel.g, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 2u] = (u8)((float)clamp(pixel.b, 0.0f, 1.0f) * 255.0f);
      data[coord.x * bpc + coord.y * size.x * bpc + 3u] = (u8)((float)clamp(pixel.a, 0.0f, 1.0f) * 255.0f);
      break;
    }
    // clang-format on
    default: ASSERT_PANIC(false && "unsupported format");
    }
  }
  vec4 sample(vec2 uv) const {
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
  }
  Image2D_Raw downsample() const {
    u32 new_width  = MAX(1, width >> 1);
    u32 new_height = MAX(1, height >> 1);

    Image2D_Raw out;
    out.init(new_width, new_height, format, NULL);

    ito(new_height) {
      jto(new_width) {
        float4 p00 = load({j * 2 + 0, i * 2 + 0});
        float4 p01 = load({j * 2 + 0, i * 2 + 1});
        float4 p10 = load({j * 2 + 1, i * 2 + 0});
        float4 p11 = load({j * 2 + 1, i * 2 + 1});
        out.write({j, i}, (p00 + p01 + p10 + p11) / 4.0f);
      }
    }

    return out;
  }
};

static inline float3 safe_normalize(float3 v) {
  return v / (glm::length(v) + 1.0e-5f);
}

struct Vertex_Full {
  float3 position;
  float3 normal;
  float3 binormal;
  float3 tangent;
  float2 u0;
  float2 u1;
  float2 u2;
  float2 u3;
  u8 *   get_attribute(rd::Attriute_t type) {
    switch (type) {
    case rd::Attriute_t::POSITION: return (u8 *)&position;
    case rd::Attriute_t::NORMAL: return (u8 *)&normal;
    case rd::Attriute_t::BINORMAL: return (u8 *)&binormal;
    case rd::Attriute_t::TANGENT: return (u8 *)&tangent;
    case rd::Attriute_t::TEXCOORD0: return (u8 *)&u0;
    case rd::Attriute_t::TEXCOORD1: return (u8 *)&u1;
    case rd::Attriute_t::TEXCOORD2: return (u8 *)&u2;
    case rd::Attriute_t::TEXCOORD3: return (u8 *)&u3;
    default: TRAP;
    }
  }
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
struct PBR_Material {
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
  u32            size;
};

// Copy pasted from meshoptimizer

struct Meshlet {
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
  union {
    struct {
      i8 cone_axis_s8[3];
      i8 cone_cutoff_s8;
    };
    u32 cone_pack;
  };
};

struct Raw_Meshlets_Opaque {
  Array<u8>                  attribute_data;
  Array<u8>                  index_data;
  Array<Meshlet>             meshlets;
  InlineArray<Attribute, 16> attributes;
  u32                        num_vertices;
  u32                        num_indices;
  u32                        get_attribute_size(u32 index) const {
    return attributes[index].size * num_vertices;
  }
  void init() {
    MEMZERO(*this);
    attributes.init();
    attribute_data.init();
    meshlets.init();
    index_data.init();
  }
  void release() {
    attributes.release();
    attribute_data.release();
    meshlets.release();
    index_data.release();
  }
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
  bool                       is_compatible(Raw_Mesh_Opaque const &that) const {
    if (attributes.size != that.attributes.size ||
        index_type != that.index_type)
      return false;
    ito(attributes.size) {
      if (attributes[i].format != that.attributes[i].format ||
          attributes[i].stride != that.attributes[i].stride ||
          attributes[i].size != that.attributes[i].size ||
          attributes[i].type != that.attributes[i].type || false

      )
        return false;
    }
    return true;
  }
  void sort_attributes() {
    quicky_sort(attributes.elems, attributes.size,
                [](Attribute const &a, Attribute const &b) {
                  return (u32)a.type < (u32)b.type;
                });
  }
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

  u8 *get_attribute_data(u32 attrib_index, u32 vertex_index) {
    return &attribute_data[attributes[attrib_index].offset +
                           attributes[attrib_index].stride * vertex_index];
  }

  u32 get_attribute_size(u32 index) const {
    return attributes[index].size * num_vertices;
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
  u32 get_bytes_per_index() const {
    if (index_type == rd::Index_t::UINT16) {
      return 2;
    } else if (index_type == rd::Index_t::UINT32) {
      return 4;
    } else {
      TRAP;
    }
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

struct BVH_Node {
  // Bit layout:
  // +-------------------------+
  // | 32 31 30 29 28 27 26 25 |
  // | 24 23 22 21 20 19 18 17 |
  // | 16 15 14 13 12 11 10 9  |
  // | 8  7  6  5  4  3  2  1  |
  // +-------------------------+
  // +--------------+
  // | [32:32] Leaf |
  // +--------------+
  // |  Leaf:
  // +->+---------------------+---------------------+
  // |  | [31:25] Item count  | [24:1] Items offset |
  // |  +---------------------+---------------------+
  // |
  // |  Branch:
  // +->+----------------------------+
  //    | [24:1]  First child offset |
  //    +----------------------------+

  // constants
  static constexpr u32 LEAF_BIT = 1 << 31;
  // Leaf flags:
  static constexpr u32 ITEMS_OFFSET_MASK  = 0xffffff;  // 24 bits
  static constexpr u32 ITEMS_OFFSET_SHIFT = 0;         // low bits
  static constexpr u32 NUM_ITEMS_MASK     = 0b1111111; // 7 bits
  static constexpr u32 NUM_ITEMS_SHIFT    = 24;        // after first 24 bits
  static constexpr u32 MAX_ITEMS          = 16;        // max items
  // Node flags:
  static constexpr u32 FIRST_CHILD_MASK  = 0xffffff;
  static constexpr u32 FIRST_CHILD_SHIFT = 0;
  static constexpr u32 MAX_DEPTH         = 20;
  static constexpr f32 EPS               = 1.0e-3f;

  float3 min;
  float3 max;
  u32    flags;

  bool intersects(float3 tmin, float3 tmax) {
    return                 //
        tmax.x >= min.x && //
        tmin.x <= max.x && //
        tmax.y >= min.y && //
        tmin.y <= max.y && //
        tmax.z >= min.z && //
        tmin.z <= max.z && //
        true;
  }
  bool inside(float3 tmin) {
    return                 //
        tmin.x >= min.x && //
        tmin.x <= max.x && //
        tmin.y >= min.y && //
        tmin.y <= max.y && //
        tmin.z >= min.z && //
        tmin.z <= max.z && //
        true;
  }
  bool intersects_ray(float3 ro, float3 rd, float min_t) {
    if (inside(ro)) return true;
    float3 invd = 1.0f / rd;
    float  dx_n = (min.x - ro.x) * invd.x;
    float  dy_n = (min.y - ro.y) * invd.y;
    float  dz_n = (min.z - ro.z) * invd.z;
    float  dx_f = (max.x - ro.x) * invd.x;
    float  dy_f = (max.y - ro.y) * invd.y;
    float  dz_f = (max.z - ro.z) * invd.z;
    float  nt   = MAX3(MIN(dx_n, dx_f), MIN(dy_n, dy_f), MIN(dz_n, dz_f));
    float  ft   = MIN3(MAX(dx_n, dx_f), MAX(dy_n, dy_f), MAX(dz_n, dz_f));
    if (nt > min_t || nt > ft - EPS) return false;
    return true;
  }
  bool intersects_ray(float3 ro, float3 rd) {
    if (inside(ro)) return true;
    float3 invd = 1.0f / rd;
    float  dx_n = (min.x - ro.x) * invd.x;
    float  dy_n = (min.y - ro.y) * invd.y;
    float  dz_n = (min.z - ro.z) * invd.z;
    float  dx_f = (max.x - ro.x) * invd.x;
    float  dy_f = (max.y - ro.y) * invd.y;
    float  dz_f = (max.z - ro.z) * invd.z;
    float  nt   = MAX3(MIN(dx_n, dx_f), MIN(dy_n, dy_f), MIN(dz_n, dz_f));
    float  ft   = MIN3(MAX(dx_n, dx_f), MAX(dy_n, dy_f), MAX(dz_n, dz_f));
    if (nt > ft - EPS) return false;
    return true;
  }
  void init_leaf(float3 min, float3 max, u32 offset) {
    flags = LEAF_BIT;
    ASSERT_DEBUG(offset <= ITEMS_OFFSET_MASK);
    flags |= ((offset << ITEMS_OFFSET_SHIFT));
    this->min = min;
    this->max = max;
  }
  void init_branch(float3 min, float3 max, BVH_Node *child) {
    ptrdiff_t diff = ((u8 *)child - (u8 *)this) / sizeof(BVH_Node);
    ASSERT_DEBUG(diff > 0 && diff < FIRST_CHILD_MASK);
    flags     = ((u32)diff << FIRST_CHILD_SHIFT);
    this->min = min;
    this->max = max;
  }
  bool is_leaf() { return (flags & LEAF_BIT) == LEAF_BIT; }
  u32  num_items() { return ((flags >> NUM_ITEMS_SHIFT) & NUM_ITEMS_MASK); }
  u32  items_offset() {
    return ((flags >> ITEMS_OFFSET_SHIFT) & ITEMS_OFFSET_MASK);
  }
  BVH_Node *first_child() {
    return this + (((flags >> FIRST_CHILD_SHIFT) & FIRST_CHILD_MASK));
  }
  void set_num_items(u32 num) {
    ASSERT_DEBUG(num <= NUM_ITEMS_MASK);
    flags &= ~(NUM_ITEMS_MASK << NUM_ITEMS_SHIFT);
    flags |= (num << NUM_ITEMS_SHIFT);
  }
  void add_item() { set_num_items(num_items() + 1); }
  bool is_full() { return num_items() == MAX_ITEMS - 1; }
};

class Node {
  protected:
  u32                 id;
  float4x4            transform_cache;
  Node *              parent;
  Array<Node *>       children;
  float3              aabb_min;
  float3              aabb_max;
  inline_string<0x10> name;

  static u32 gen_id() {
    static u32 _id = 1;
    return _id++;
  }

  void init(string_ref name) {
    id = gen_id();
    if (name.len == 0) {
      char buf[0x10];
      snprintf(buf, sizeof(buf), "node_%i", id);
      this->name.init(stref_s(buf));
    } else
      this->name.init(name);
    children.init();
    parent          = NULL;
    scale           = float3(1.0f, 1.0f, 1.0f);
    offset          = float3(0.0f, 0.0f, 0.0f);
    aabb_min        = float3(0.0f, 0.0f, 0.0f);
    aabb_max        = float3(0.0f, 0.0f, 0.0f);
    rotation        = quat();
    transform_cache = float4x4(1.0f);
  }

  public:
  float3 offset;
  quat   rotation;
  float3 scale;

  Node *rename(string_ref name) {
    this->name.init(name);
    return this;
  }

  static Node *create(string_ref name) {
    Node *out = new Node;
    out->init(name);
    return out;
  }
  static u64 ID() {
    static char p;
    return (u64)(intptr_t)&p;
  }
  virtual u64  get_type_id() const { return ID(); }
  string_ref   get_name() const { return name.ref(); }
  u32          get_id() const { return id; }
  virtual void release() {
    ito(children.size) children[i]->release();
    children.release();
    delete this;
  }
  void set_parent(Node *node) { parent = node; }
  void add_child(Node *node) {
    children.push(node);
    node->set_parent(node);
  }
  void update_cache(float4x4 const &parent = float4x4(1.0f)) {
    transform_cache = parent * get_transform();
  }
  float4x4 get_transform() {
    return glm::translate(float4x4(1.0f), offset) * (float4x4)rotation *
           glm::scale(float4x4(1.0f), scale);
  }
  float4x4 get_cofactor() {
    mat4 out{};
    mat4 transform = get_transform();
    cofactor(&transform[0][0], &out[0][0]);
  }
  virtual void dump(u32 indent = 0) const {
    ito(indent) fprintf(stdout, " ");
    string_ref n = get_name();
    fprintf(stdout, "%.*s\n", STRF(n));
    ito(children.size) children[i]->dump(indent + 2);
  }
  void set_aabb(float3 aabb_min, float3 aabb_max) {
    this->aabb_min = aabb_min;
    this->aabb_max = aabb_max;
  }
  Array<Node *> const &get_children() const { return children; }
  virtual ~Node() {}
};

template <typename T> struct BVH_Helper {
  float3      min;
  float3      max;
  Array<T>    items;
  BVH_Helper *left;
  BVH_Helper *right;
  bool        is_leaf;
  void        init() {
    MEMZERO(*this);
    items.init();
    min     = float3(1.0e10f, 1.0e10f, 1.0e10f);
    max     = float3(-1.0e10f, -1.0e10f, -1.0e10f);
    is_leaf = true;
  }
  void release() {
    if (left != NULL) left->release();
    if (right != NULL) right->release();
    items.release();
    MEMZERO(*this);
    delete this;
  }
  void reserve(size_t size) { items.reserve(size); }
  void push(T const &item) {
    items.push(item);
    float3 tmin, tmax;
    get_aabb(item, tmin, tmax);
    ito(3) min[i] = MIN(min[i], tmin[i]);
    ito(3) max[i] = MAX(max[i], tmax[i]);
  }
  u32 split(u32 max_items, u32 depth = 0) {
    ASSERT_DEBUG(depth < BVH_Node::MAX_DEPTH);
    if (items.size > max_items && depth < BVH_Node::MAX_DEPTH) {
      left = new BVH_Helper;
      left->init();
      left->reserve(items.size / 2);
      right = new BVH_Helper;
      right->init();
      right->reserve(items.size / 2);
      struct Sorting_Node {
        u32   id;
        float val;
      };
      {
        TMP_STORAGE_SCOPE;
        u32           num_items = items.size;
        Sorting_Node *sorted_dims[6];
        ito(6) sorted_dims[i] =
            (Sorting_Node *)tl_alloc_tmp(sizeof(Sorting_Node) * num_items);
        T *items = items.ptr;
        ito(num_items) {
          float3 tmin, tmax;
          get_aabb(items[i], tmin, tmax);
          jto(3) {
            sorted_dims[j][i].val     = tmin[j];
            sorted_dims[j][i].id      = i;
            sorted_dims[j + 3][i].val = tmax[j];
            sorted_dims[j + 3][i].id  = i;
          }
        }
        ito(6) quicky_sort(sorted_dims[i], num_items,
                           [](Sorting_Node const &a, Sorting_Node const &b) {
                             return a.val < b.val;
                           });
        float max_dim_diff = 0.0f;
        u32   max_dim_id   = 0;
        u32   last_item    = num_items - 1;
        ito(3) {
          // max - min
          float diff =
              sorted_dims[i + 3][last_item].val - sorted_dims[i][0].val;
          if (diff > max_dim_diff) {
            max_dim_diff = diff;
            max_dim_id   = i;
          }
        }
        u32 split_index = (last_item + 1) / 2;
        ito(num_items) {
          u32 item_id = sorted_dims[max_dim_id][i].id;
          T   item    = items[item_id];
          if (i < split_index) {
            left->push(item);
          } else {
            right->push(item);
          }
        }
      }
      is_leaf = false;
      items.release();
      u32 cnt = left->split(max_items, depth + 1);
      cnt += right->split(max_items, depth + 1);
      return cnt + 1;
    }
    return 1;
  }
};

static_assert(sizeof(BVH_Node) == 28, "Blamey!");

template <typename T> struct BVH {
  Array<T>        item_pool;
  Array<BVH_Node> node_pool;
  BVH_Node *      root;

  void gen(BVH_Node *node, BVH_Helper<T> *hnode) {
    ASSERT_ALWAYS(node != NULL);
    ASSERT_ALWAYS(hnode != NULL);
    if (hnode->is_leaf) {
      ASSERT_DEBUG(hnode->items.size != 0);
      u32 item_offset = alloc_item_chunk();
      node->init_leaf(hnode->min, hnode->max, item_offset);
      ASSERT_DEBUG(hnode->items.size <= BVH_Node::MAX_ITEMS);
      node->set_num_items(hnode->items.size);
      T *items = item_pool.at(node->items_offset());
      ito(hnode->items.size) { items->store(i, hnode->items[i]); }
    } else {
      BVH_Node *children = node_pool.alloc(2);
      node->init_branch(hnode->min, hnode->max, children);
      gen(children + 0, hnode->left);
      gen(children + 1, hnode->right);
    }
  }
  void init(T *items, u32 num_items) { //
    BVH_Helper *hroot = new BVH_Helper;
    hroot->init();
    hroot->reserve(num_items);
    defer(hroot->release());
    ito(num_items) { hroot->push(items[i]); }
    u32 ncnt = hroot->split(BVH_Node::MAX_ITEMS);
    item_pool.init();
    node_pool.init();
    item_pool.reserve(num_items * 4);
    node_pool.reserve(ncnt);
    root = node_pool.alloc(1);
    gen(root, hroot);
  }
  u32 alloc_item_chunk() {
    T *new_chunk = item_pool.alloc(1);
    MEMZERO(*new_chunk);
    T *item_root = item_pool.at(0);
    return (u32)(((u8 *)new_chunk - (u8 *)item_root) / sizeof(T));
  }
  void release() {
    item_pool.release();
    node_pool.release();
  }
  template <typename F> void traverse(float3 ro, float3 rd, F fn) {
    if (!root->intersects_ray(ro, rd)) return;
    traverse(root, ro, rd, fn);
  }
  template <typename F>
  void traverse(BVH_Node *node, float3 ro, float3 rd, F fn) {
    if (node->is_leaf()) {
      T * items     = item_pool.at(node->items_offset());
      u32 num_items = node->num_items();
      ASSERT_ALWAYS(num_items <= vfloat3::WIDTH);
      fn(*items);
    } else {
      BVH_Node *children = node->first_child();
      BVH_Node *left     = children + 0;
      BVH_Node *right    = children + 1;
      if (left->intersects_ray(ro, rd)) traverse(left, ro, rd, fn);
      if (right->intersects_ray(ro, rd)) traverse(right, ro, rd, fn);
    }
  }
};

struct Primitive {
  Raw_Mesh_Opaque     mesh;
  Raw_Meshlets_Opaque meshlets;
  PBR_Material        material;
  void                init() {
    mesh.init();
    material.init();
    meshlets.init();
  }
  void release() {
    mesh.release();
    meshlets.release();
  }
};

class MeshNode : public Node {
  protected:
  Array<Primitive> primitives;
  void             init(string_ref name) {
    Node::init(name);
    primitives.init();
  }

  public:
  static MeshNode *create(string_ref name) {
    MeshNode *out = new MeshNode;
    out->init(name);
    return out;
  }
  static u64 ID() {
    static char p;
    return (u64)(intptr_t)&p;
  }
  u64  get_type_id() const override { return ID(); }
  void add_primitive(Raw_Mesh_Opaque &mesh, Raw_Meshlets_Opaque &meshlets,
                     PBR_Material &mat) {
    if (primitives.size != 0) {
      ASSERT_ALWAYS(primitives[0].mesh.is_compatible(mesh));
    }
    Primitive p;
    p.init();
    p.mesh     = mesh;
    p.material = mat;
    p.meshlets = meshlets;
    primitives.push(p);
  }
  void release() override {
    ito(primitives.size) primitives[i].release();
    primitives.release();
    Node::release();
  }
  Array<Primitive> const &get_primitives() const { return primitives; }
};

template <typename T> static bool isa(Node *node) {
  return node->get_type_id() == T::ID();
}

class IFactory {
  public:
  virtual Node *    add_node(string_ref name)  = 0;
  virtual MeshNode *add_mesh(string_ref name)  = 0;
  virtual u32       add_image(Image2D_Raw img) = 0;
};

Node *      load_gltf_pbr(IFactory *factory, string_ref filename);
Image2D_Raw load_image(string_ref filename,
                       rd::Format format = rd::Format::RGBA8_SRGBA);
#endif // SCENE