#ifndef SCENE_HPP
#define SCENE_HPP

#include "rendering.hpp"
#include "script.hpp"
#include "utils.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

#ifdef VULKII_EMBREE
#  include <embree3/rtcore.h>
#  include <embree3/rtcore_builder.h>
#  include <embree3/rtcore_scene.h>
#endif

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
#  define ALIGN16 __attribute__((aligned(16)))
#else
#  define ALIGN16 __declspec(align(16))
#endif

typedef i32 ALIGN16   ai32;
typedef u32 ALIGN16   au32;
typedef float ALIGN16 af32;
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

static constexpr float PI                  = 3.1415926f;
static constexpr float TWO_PI              = 6.2831852f;
static constexpr float FOUR_PI             = 12.566370f;
static constexpr float INV_PI              = 0.3183099f;
static constexpr float INV_TWO_PI          = 0.1591549f;
static constexpr float INV_FOUR_PI         = 0.0795775f;
static constexpr float DIELECTRIC_SPECULAR = 0.04f;

// https://github.com/graphitemaster/normals_revisited
static float minor(const float m[16], int r0, int r1, int r2, int c0, int c1, int c2) {
  return m[4 * r0 + c0] * (m[4 * r1 + c1] * m[4 * r2 + c2] - m[4 * r2 + c1] * m[4 * r1 + c2]) -
         m[4 * r0 + c1] * (m[4 * r1 + c0] * m[4 * r2 + c2] - m[4 * r2 + c0] * m[4 * r1 + c2]) +
         m[4 * r0 + c2] * (m[4 * r1 + c0] * m[4 * r2 + c1] - m[4 * r2 + c0] * m[4 * r1 + c1]);
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
    i = (i32)std::floorf((float)i / (float)base);
    x /= base;
  }
  return v;
}

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
  float3 polar_to_cartesian(float sinTheta, float cosTheta, float sinPhi, float cosPhi) {
    return float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
  }
  // Z is up here
  float3 uniform_sample_cone(float cos_theta_max, float3 xbasis, float3 ybasis, float3 zbasis) {
    vec2   rand     = vec2(rand_unit_float(), rand_unit_float());
    float  cosTheta = (1.0f - rand.x) + rand.x * cos_theta_max;
    float  sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    float  phi      = rand.y * PI * 2.0f;
    float3 samplev  = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
  }

  float3 rand_sphere_center() {
    vec2   rand     = vec2(rand_unit_float(), rand_unit_float());
    float  cosTheta = rand.x * 2.0f - 1.0f;
    float  sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    float  phi      = rand.y * PI * 2.0f;
    float3 samplev  = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    float  r        = rand_unit_float();
    return r * samplev;
  }

  float3 rand_sphere_center_r2() {
    vec2   rand     = vec2(rand_unit_float(), rand_unit_float());
    float  cosTheta = rand.x * 2.0f - 1.0f;
    float  sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    float  phi      = rand.y * PI * 2.0f;
    float3 samplev  = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    float  r        = rand_unit_float();
    return r * r * samplev;
  }

  float3 rand_sphere_center_r3() {
    vec2   rand     = vec2(rand_unit_float(), rand_unit_float());
    float  cosTheta = rand.x * 2.0f - 1.0f;
    float  sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    float  phi      = rand.y * PI * 2.0f;
    float3 samplev  = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    float  r        = rand_unit_float();
    return r * r * r * samplev;
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

  float3 sample_lambert_BRDF(float3 N) { return normalize(N + rand_unit_sphere()); }

  vec2 random_halton() {
    f32 u = halton(halton_id + 1, 2);
    f32 v = halton(halton_id + 1, 3);
    halton_id++;
    return vec2(u, v);
  }

  static float3 SampleHemisphere_Cosinus(float2 xi) {
    float phi      = xi.y * 2.0f * PI;
    float cosTheta = std::sqrtf(1.0f - xi.x);
    float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);

    return float3(std::cosf(phi) * sinTheta, std::sinf(phi) * sinTheta, cosTheta);
  }

  private:
  PCG pcg;
  u32 halton_id = 0;
};

struct Image2D {
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

  static Image2D *create(u32 width, u32 height, rd::Format format, u8 *data) {
    Image2D *i = new Image2D;
    i->init(width, height, format, data);
    return i;
  }

  u32  get_size_in_bytes() const { return get_bpp() * width * height; }
  void release() {
    if (data != NULL) tl_free(data);
    MEMZERO(*this);
    delete this;
  }
  static u32 get_bpp(rd::Format format) {
    switch (format) {
    case rd::Format::RGBA8_UNORM:
    case rd::Format::RGBA8_SRGBA: return 4u;
    case rd::Format::RGB32_FLOAT: return 12u;
    case rd::Format::RGBA32_FLOAT: return 16u;
    default: ASSERT_PANIC(false && "unsupported format");
    }
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
      return *(f32 *)&data[coord.x * bpc + coord.y * size.x * bpc + component * 4u];
    };
    int2 size = int2(width, height);
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
      return vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f, float(a) / 255.0f);
    }
    case rd::Format::RGBA8_SRGBA: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];

      auto out = vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f, float(a) / 255.0f);
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
    int2 size = int2(width, height);
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
    vec2  fract     = vec2(suv.x - std::floor(suv.x), suv.y - std::floor(suv.y));
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
  Image2D downsample() const {
    u32 new_width  = MAX(1, width >> 1);
    u32 new_height = MAX(1, height >> 1);

    Image2D out;
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
  u32        get_num_mip_levels() const { return get_num_mip_levels(width, height); }
  static u32 get_num_mip_levels(u32 width, u32 height) {
    u32 mip_levels = 0;
    u32 w          = width;
    u32 h          = height;
    while (w || h) {
      w /= 2;
      h /= 2;
      mip_levels++;
    }
    return MAX(1, mip_levels);
  }
};

static inline float3 safe_normalize(float3 v) { return v / (glm::length(v) + 1.0e-5f); }

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

struct u16_face {
  union {
    struct {
      uint16_t v0, v1, v2;
    };
    struct {
      uint16_t arr[3];
    };
  };
  uint16_t  operator[](size_t i) const { return arr[i]; }
  uint16_t &operator[](size_t i) { return arr[i]; }
};
struct Raw_Mesh_3p16i {
  Array<float3>   positions;
  Array<u16_face> indices;
  void            init() {
    positions.init();
    indices.init();
  }
  void release() {
    positions.release();
    indices.release();
  }
};

static Raw_Mesh_3p16i subdivide_cylinder(uint32_t level, float radius, float length) {
  Raw_Mesh_3p16i out;
  out.init();
  level += 4;
  float step = PI * 2.0f / level;
  out.positions.resize(level * 2);
  for (u32 i = 0; i < level; i++) {
    float angle              = step * i;
    out.positions[i]         = {radius * std::cos(angle), radius * std::sin(angle), 0.0f};
    out.positions[i + level] = {radius * std::cos(angle), radius * std::sin(angle), length};
  }
  for (u32 i = 0; i < level; i++) {
    out.indices.push(u16_face{(u16)i, (u16)(i + level), (u16)((i + 1) % level)});
    out.indices.push(
        u16_face{(u16)((i + 1) % level), (u16)(i + level), (u16)(((i + 1) % level) + level)});
  }
  return out;
}

static Raw_Mesh_3p16i subdivide_icosahedron(uint32_t level) {
  Raw_Mesh_3p16i out;
  out.init();
  static float const  X                           = 0.5257311f;
  static float const  Z                           = 0.8506508f;
  static float3 const g_icosahedron_positions[12] = {
      {-X, 0.0, Z}, {X, 0.0, Z},   {-X, 0.0, -Z}, {X, 0.0, -Z}, {0.0, Z, X},  {0.0, Z, -X},
      {0.0, -Z, X}, {0.0, -Z, -X}, {Z, X, 0.0},   {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}};
  static u16_face const g_icosahedron_indices[20] = {
      {1, 4, 0}, {4, 9, 0},  {4, 5, 9},  {8, 5, 4},  {1, 8, 4},  {1, 10, 8}, {10, 3, 8},
      {8, 3, 5}, {3, 2, 5},  {3, 7, 2},  {3, 10, 7}, {10, 6, 7}, {6, 11, 7}, {6, 0, 11},
      {6, 1, 0}, {10, 1, 6}, {11, 0, 9}, {2, 11, 9}, {5, 2, 9},  {11, 2, 7}};
  for (auto p : g_icosahedron_positions) {
    out.positions.push(p);
  }
  for (auto i : g_icosahedron_indices) {
    out.indices.push(i);
  }
  auto subdivide = [](Raw_Mesh_3p16i const &in) {
    Raw_Mesh_3p16i out;
    out.init();
    Hash_Table<Pair<uint16_t, uint16_t>, uint16_t> lookup;
    lookup.init();
    defer(lookup.release());
    auto get_or_insert = [&](uint16_t i0, uint16_t i1) {
      Pair<uint16_t, uint16_t> key{i0, i1};
      if (key.first > key.second) swap(key.first, key.second);

      if (!lookup.contains(key)) {
        lookup.insert(key, (u32)out.positions.size);
        auto v0_x = out.positions[i0].x;
        auto v1_x = out.positions[i1].x;
        auto v0_y = out.positions[i0].y;
        auto v1_y = out.positions[i1].y;
        auto v0_z = out.positions[i0].z;
        auto v1_z = out.positions[i1].z;

        auto mid_point  = float3{(v0_x + v1_x) / 2.0f, (v0_y + v1_y) / 2.0f, (v0_z + v1_z) / 2.0f};
        auto add_vertex = [&](float3 p) {
          float length = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
          out.positions.push(float3{p.x / length, p.y / length, p.z / length});
        };
        add_vertex(mid_point);
      }

      return lookup.get(key);
    };
    out.positions = in.positions;
    ito(in.indices.size) {
      auto &   face = in.indices[i];
      u16_face mid;

      for (size_t edge = 0; edge < 3; ++edge) {
        mid[edge] = get_or_insert(face[edge], face[(edge + 1) % 3]);
      }

      out.indices.push(u16_face{face[0], mid[0], mid[2]});
      out.indices.push(u16_face{face[1], mid[1], mid[0]});
      out.indices.push(u16_face{face[2], mid[2], mid[1]});
      out.indices.push(u16_face{mid[0], mid[1], mid[2]});
    }
    return out;
  };
  for (uint i = 0; i < level; i++) {
    out = subdivide(out);
  }
  return out;
}

static Raw_Mesh_3p16i subdivide_cone(uint32_t level, float radius, float length) {
  Raw_Mesh_3p16i out;
  out.init();
  level += 4;
  float step = PI * 2.0f / level;
  out.positions.resize(level * 2 + 2);
  out.positions[0] = {0.0f, 0.0f, 0.0f};
  out.positions[1] = {0.0f, 0.0f, length};
  for (u32 i = 0; i < level; i++) {
    float angle          = step * i;
    out.positions[i + 2] = {radius * std::cos(angle), radius * std::sin(angle), 0.0f};
  }
  for (u32 i = 0; i < level; i++) {
    out.indices.push(u16_face{(u16)(i + 2), (u16)(2 + (i + 1) % level), (u16)0});
    out.indices.push(u16_face{(u16)(i + 2), (u16)(2 + (i + 1) % level), (u16)1});
  }
  return out;
}

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
  size_t         offset;
  size_t         stride;
  size_t         size;
};

struct Meshlet {
  u32 vertex_offset;
  u32 index_offset;
  u32 index_count;
  u32 vertex_count;
  //// (x, y, z, radius)
  // float4 sphere;
  //// (x, y, z, _)
  // float4 cone_apex;
  //// (x, y, z, cutoff)
  // float4 cone_axis_cutoff;
  // union {
  //  struct {
  //    i8 cone_axis_s8[3];
  //    i8 cone_cutoff_s8;
  //  };
  //  u32 cone_pack;
  //};
};

class Raw_Meshlets_Opaque {
  private:
  ~Raw_Meshlets_Opaque() = default;

  public:
  Raw_Meshlets_Opaque() { MEMZERO(*this); }

  Array<u8>      attribute_data;
  Array<u8>      index_data;
  Array<Meshlet> meshlets;
  // Array<u32>                 meshlet_vertices;
  InlineArray<Attribute, 16> attributes;
  u32                        num_vertices;
  // u32                        num_indices;
  float3 &fetch_position(u32 index) const {
    ito(attributes.size) {
      switch (attributes[i].type) {

      case rd::Attriute_t ::POSITION:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        // float3 pos;
        // memcpy(&pos, attribute_data.at(index * attributes[i].stride + attributes[i].offset), 12);
        // return pos;
        return *(float3 *)attribute_data.at(index * attributes[i].stride + attributes[i].offset);
      default: break;
      }
    }
    TRAP;
  }
  u32  get_attribute_size(u32 index) const { return attributes[index].size * num_vertices; }
  void init() {
    MEMZERO(*this);
    attributes.init();
    attribute_data.init();
    meshlets.init();
    index_data.init();
    // meshlet_vertices.init();
  }
  void release() {
    attributes.release();
    attribute_data.release();
    meshlets.release();
    index_data.release();
    // meshlet_vertices.release();
    delete this;
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
  u32                        id;

  static u32 gen_id() {
    static u32 _id = 1;
    return _id++;
  }
  u32 get_vertex_size() const {
    u32 size = 0;
    ito(attributes.size) { size += attributes[i].size; }
    return size;
  }
  void write_vertex(void *dst, u32 index) const {
    u32 offset = 0;
    ito(attributes.size) {
      memcpy((u8 *)dst + offset,
             &attribute_data[0] + attributes[i].offset + attributes[i].stride * index,
             attributes[i].size);
      offset += attributes[i].size;
    }
  }
  u32 write_attribute(void *dst, u32 vertex_index, u32 attribute_index) const {
    memcpy((u8 *)dst,
           &attribute_data[0] + attributes[attribute_index].offset +
               attributes[attribute_index].stride * vertex_index,
           attributes[attribute_index].size);
    return attributes[attribute_index].size;
  }
  void flatten(void *dst) const {
    u32 vertex_stride = get_vertex_size();
    ito(num_indices) {
      u32 index = fetch_index(i);
      write_vertex((u8 *)dst + vertex_stride * i, index);
    }
  }
  void interleave(void *dst) const {
    u32 vertex_stride = get_vertex_size();
    ito(num_vertices) { write_vertex((u8 *)dst + vertex_stride * i, i); }
  }
  void interleave() {
    Array<u8> dst;
    dst.init();
    dst.resize(attribute_data.size);
    interleave(dst.ptr);
    attribute_data.release();
    attribute_data  = dst;
    u32 offset      = 0;
    u32 vertex_size = get_vertex_size();
    ito(attributes.size) {
      attributes[i].stride = vertex_size;
      attributes[i].offset = offset;
      offset += attributes[i].size;
    }
  }
  void deinterleave() {
    Array<u8> dst;
    dst.init();
    dst.resize(attribute_data.size);
    InlineArray<size_t, 16> attribute_offsets;
    InlineArray<size_t, 16> attribute_sizes;
    InlineArray<size_t, 16> attribute_cursors;
    attribute_offsets.init();
    attribute_sizes.init();
    attribute_cursors.init();
    ito(attributes.size) { attribute_sizes[i] = num_vertices * attributes[i].size; }
    u32 total_mem = 0;
    ito(attributes.size) {
      jto(i) attribute_offsets[i] += attribute_sizes[j];
      total_mem = (u32)attribute_offsets[i] + (u32)attribute_sizes[i];
    }
    ito(num_vertices) {
      jto(attributes.size) {
        attribute_cursors[j] +=
            write_attribute(dst.ptr + attribute_offsets[j] + attribute_cursors[j], i, j);
      }
    }
    ito(attributes.size) {
      attributes[i].stride = attributes[i].size;
      attributes[i].offset = (u32)attribute_offsets[i];
    }
    attribute_data.release();
    attribute_data = dst;
  }
  bool is_compatible(Raw_Mesh_Opaque const &that) const {
    if (attributes.size != that.attributes.size || index_type != that.index_type) return false;
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
                [](Attribute const &a, Attribute const &b) { return (u32)a.type < (u32)b.type; });
  }
  void init() {
    MEMZERO(*this);
    attributes.init();
    attribute_data.init();
    index_data.init();
    id = gen_id();
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
  u32     get_attribute_size(u32 index) const { return attributes[index].size * num_vertices; }
  float3 &fetch_position(u32 index) const {
    ito(attributes.size) {
      switch (attributes[i].type) {

      case rd::Attriute_t ::POSITION:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        // float3 pos;
        // memcpy(&pos, attribute_data.at(index * attributes[i].stride + attributes[i].offset), 12);
        // return pos;
        return *(float3 *)attribute_data.at(index * attributes[i].stride + attributes[i].offset);
      default: break;
      }
    }
    TRAP;
  }
  Vertex_Full fetch_vertex(u32 index) const {
    Vertex_Full v;
    MEMZERO(v);
    ito(attributes.size) {
      switch (attributes[i].type) {
      case rd::Attriute_t ::NORMAL:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.normal, attribute_data.at(index * attributes[i].stride + attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::BINORMAL:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.binormal, attribute_data.at(index * attributes[i].stride + attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::TANGENT:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGBA32_FLOAT);
        memcpy(&v.tangent, attribute_data.at(index * attributes[i].stride + attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::POSITION:
        ASSERT_PANIC(attributes[i].format == rd::Format::RGB32_FLOAT);
        memcpy(&v.position, attribute_data.at(index * attributes[i].stride + attributes[i].offset),
               12);
        break;
      case rd::Attriute_t ::TEXCOORD0:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u0, attribute_data.at(index * attributes[i].stride + attributes[i].offset), 8);
        break;
      case rd::Attriute_t ::TEXCOORD1:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u1, attribute_data.at(index * attributes[i].stride + attributes[i].offset), 8);
        break;
      case rd::Attriute_t ::TEXCOORD2:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u2, attribute_data.at(index * attributes[i].stride + attributes[i].offset), 8);
        break;
      case rd::Attriute_t ::TEXCOORD3:
        ASSERT_PANIC(attributes[i].format == rd::Format::RG32_FLOAT);
        memcpy(&v.u3, attribute_data.at(index * attributes[i].stride + attributes[i].offset), 8);
        break;
      default: TRAP;
      }
    }
    return v;
  }

  u32 fetch_index(u32 index) const {
    if (index_type == rd::Index_t::UINT16) {
      return (u32) * (u16 *)index_data.at(2 * index);
    } else {
      return (u32) * (u32 *)index_data.at(4 * index);
    }
  }

  Tri_Index get_tri_index(u32 id) const {
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
  Triangle_Full fetch_triangle(u32 id) const {
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
    vertex.normal   = safe_normalize(v0.normal * k0 + v1.normal * k1 + v2.normal * k2);
    vertex.position = v0.position * k0 + v1.position * k1 + v2.position * k2;
    vertex.tangent  = safe_normalize(v0.tangent * k0 + v1.tangent * k1 + v2.tangent * k2);
    vertex.binormal = safe_normalize(v0.binormal * k0 + v1.binormal * k1 + v2.binormal * k2);
    vertex.u0       = v0.u0 * k0 + v1.u0 * k1 + v2.u0 * k2;
    vertex.u1       = v0.u1 * k0 + v1.u1 * k1 + v2.u1 * k2;
    vertex.u2       = v0.u2 * k0 + v1.u2 * k1 + v2.u2 * k2;
    vertex.u3       = v0.u3 * k0 + v1.u3 * k1 + v2.u3 * k2;
    return vertex;
  }
};

class Type {
  public:
  virtual char const *getName()   = 0;
  virtual Type *      getParent() = 0;
};

#define DECLARE_TYPE(X, P)                                                                         \
  class X##Type : public Type {                                                                    \
public:                                                                                            \
    char const *getName() override { return #X; }                                                  \
    Type *      getParent() override { return P::get_type(); }                                     \
  };                                                                                               \
  static Type *get_type() {                                                                        \
    static X##Type t;                                                                              \
    return &t;                                                                                     \
  }                                                                                                \
  Type *getType() override { return get_type(); }

class Typed {
  public:
  static Type *              get_type() { return NULL; }
  virtual Type *             getType() = 0;
  template <typename T> bool isa() {
    if (this == NULL) return false;
    Type *ty = getType();
    while (ty) {
      if (ty == T::get_type()) return true;
      ty = ty->getParent();
    }
    return false;
  }
  template <typename T> T *dyn_cast() {
    if (this == NULL) return NULL;
    Type *ty = getType();
    while (ty) {
      if (ty == T::get_type()) return (T *)this;
      ty = ty->getParent();
    }
    return NULL;
  }
};

struct AABB {
  float3               min;
  float3               max;
  static constexpr f32 EPS = 1.0e-6f;

  float area() {
    float3 dr = max - min;
    return 2.0f * (dr.x * dr.y + dr.x * dr.z + dr.y * dr.z);
  }

  float max_dim() { return fmaxf(max.x - min.x, fmaxf(max.y - min.y, (max.z - min.z))); }

  void init(float3 p) {
    min = p;
    max = p;
  }
  void unite(float3 p) {
    min.x = MIN(min.x, p.x);
    min.y = MIN(min.y, p.y);
    min.z = MIN(min.z, p.z);
    max.x = MAX(max.x, p.x);
    max.y = MAX(max.y, p.y);
    max.z = MAX(max.z, p.z);
  }
  void unite(AABB const &p) {
    min.x = MIN(min.x, p.min.x);
    min.y = MIN(min.y, p.min.y);
    min.z = MIN(min.z, p.min.z);
    max.x = MAX(max.x, p.max.x);
    max.y = MAX(max.y, p.max.y);
    max.z = MAX(max.z, p.max.z);
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
  bool collide(float3 ro, float3 rd, float &near_t, float &far_t) {
    if (inside(ro)) {
      near_t = 0.0f;
      return true;
    }
    float3 invd = 1.0f / rd;
    float  dx_n = (min.x - ro.x) * invd.x;
    float  dy_n = (min.y - ro.y) * invd.y;
    float  dz_n = (min.z - ro.z) * invd.z;
    float  dx_f = (max.x - ro.x) * invd.x;
    float  dy_f = (max.y - ro.y) * invd.y;
    float  dz_f = (max.z - ro.z) * invd.z;
    float  nt   = MAX3(MIN(dx_n, dx_f), MIN(dy_n, dy_f), MIN(dz_n, dz_f));
    float  ft   = MIN3(MAX(dx_n, dx_f), MAX(dy_n, dy_f), MAX(dz_n, dz_f));
    if (nt < 0.0f) return false;
    if (nt > far_t || nt > ft - EPS) return false;
    near_t = nt;
    far_t  = ft;
    return true;
  }
};

class Node : public Typed {
  public:
  class Component : public Typed {
protected:
    Node *node = NULL;

public:
    DECLARE_TYPE(Component, Typed)

    Component(Node *n) {
      node = n;
      node->addComponent(this);
    }
    virtual void update() {}
    virtual ~Component() {}
    virtual void release() {
      node->removeComponent(this);
      delete this;
    }
  };

  protected:
  u32                 id;
  float4x4            itransform_cache;
  float4x4            transform_cache;
  Node *              parent;
  Array<Node *>       children;
  Array<Component *>  components;
  AABB                aabb;
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
    components.init();
    parent          = NULL;
    scale           = float3(1.0f, 1.0f, 1.0f);
    offset          = float3(0.0f, 0.0f, 0.0f);
    aabb            = {};
    rotation        = quat();
    transform_cache = float4x4(1.0f);
  }

  public:
  DECLARE_TYPE(Node, Typed)

  float3 offset;
  quat   rotation;
  float3 scale;

  virtual Node *clone() {
    Node *new_node = create(name.ref());
    clone_into(new_node);
    return new_node;
  }

  void clone_into(Node *new_node) {
    new_node->transform_cache = this->transform_cache;
    new_node->aabb            = this->aabb;
    new_node->offset          = this->offset;
    new_node->rotation        = this->rotation;
    new_node->scale           = this->scale;
    ito(children.size) { new_node->add_child(children[i]->clone()); }
    if (parent != NULL) {
      parent->add_child(new_node);
    }
  }

  virtual void restore(List *l) {
    if (l == NULL) return;
    if (l->child) {
      restore(l->child);
      restore(l->next);
    } else {
      if (l->cmp_symbol("offset")) {
        offset.x = l->get(2)->parse_float();
        offset.y = l->get(3)->parse_float();
        offset.z = l->get(4)->parse_float();
      } else if (l->cmp_symbol("rotation")) {
        glm::vec3 euler;
        euler.x = l->get(2)->parse_float();
        euler.y = l->get(3)->parse_float();
        euler.z = l->get(4)->parse_float();
        euler *= PI / 180.0f;
        rotation = glm::quat(glm::vec3(euler.x, euler.y, euler.z));
      } else if (l->cmp_symbol("node")) {
        string_ref name = l->get(1)->symbol;
        if (Node *node = get_node(name)) {
          node->restore(l->get(2));
        }
      }
    }
  }
  Node *get_node(string_ref n) {
    if (name.ref() == n) return this;
    ito(children.size) {
      if (children[i])
        if (children[i]->name.ref() == n) return children[i];
    }
    return NULL;
  }

  virtual void save(String_Builder &sb) {
    sb.putf("(node \"\"\"%.*s\"\"\"\n", STRF(name.ref()));
    sb.putf("  (offset float3 %f %f %f)\n", offset.x, offset.y, offset.z);
    glm::vec3 euler = eulerAngles(rotation);
    euler *= 180.0f / PI;
    sb.putf("  (rotation float3 %f %f %f)\n", euler.x, euler.y, euler.z);
    ito(children.size) {
      if (children[i]) children[i]->save(sb);
    }
    sb.putf(")\n");
  }

  Node *translate(float3 dr) {
    offset += dr;
    return this;
  }

  Node *rename(string_ref name) {
    this->name.init(name);
    return this;
  }

  static Node *create(string_ref name) {
    Node *out = new Node;
    out->init(name);
    return out;
  }

  string_ref   get_name() const { return name.ref(); }
  u32          get_id() const { return id; }
  virtual void release() {
    ito(children.size) if (children[i]) children[i]->release();
    ito(components.size) if (components[i]) components[i]->release();
    children.release();
    components.release();
    delete this;
  }
  void                     addComponent(Component *c) { components.push(c); }
  void                     removeComponent(Component *c) { components.replace(c, NULL); }
  template <typename T> T *getComponent() {
    ito(components.size) {
      if (components[i]->isa<T>()) return components[i]->dyn_cast<T>();
    }
    return NULL;
  }
  void set_parent(Node *node) {
    ASSERT_ALWAYS(node != this);
    parent = node;
  }
  void add_child(Node *node) {
    children.push(node);
    node->set_parent(this);
  }
  void update_transform() {
    float4x4 parent = float4x4(1.0f);
    if (this->parent) parent = this->parent->get_transform();
    transform_cache = glm::translate(float4x4(1.0f), offset) * (float4x4)rotation *
                      glm::scale(float4x4(1.0f), scale) * parent;
    itransform_cache = inverse(transform_cache);
    // ito(children.size) {
    //  Node *child = children[i];
    //  if (child) child->update_transform(transform_cache);
    //}
  }
  float4x4 const &get_itransform() { return itransform_cache; }
  float4x4 const &get_transform() { return transform_cache; }
  float3 transform(float3 const &a) { return (transform_cache * float4(a.x, a.y, a.z, 1.0f)).xyz; }
  float4x4 get_cofactor() {
    mat4 out{};
    mat4 transform = get_transform();
    cofactor(&transform[0][0], &out[0][0]);
  }
  virtual void dump(u32 indent = 0) const {
    ito(indent) fprintf(stdout, " ");
    string_ref n = get_name();
    fprintf(stdout, "%.*s\n", STRF(n));
    ito(children.size) if (children[i]) children[i]->dump(indent + 2);
  }
  AABB                 getAABB() { return aabb; }
  Array<Node *> const &get_children() const { return children; }
  virtual void         update() {
    ito(components.size) {
      if (components[i]) {
        components[i]->update();
      }
    }
    ito(children.size) {
      if (children[i]) children[i]->update();
    }
    if (children.size)
      aabb = children[0]->getAABB();
    else
      aabb.init(offset);

    ito(children.size) {
      if (children[i]) aabb.unite(children[i]->getAABB());
    }
    update_transform();
  }
  virtual ~Node() {}
};
// Switching to embree
#if 0
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
    item.get_aabb(tmin, tmax);
    ito(3) min[i] = MIN(min[i], tmin[i]);
    ito(3) max[i] = MAX(max[i], tmax[i]);
  }
  u32 split(u32 max_items, u32 depth = 0) {
    ASSERT_DEBUG(depth < Packed_BVH_Node::MAX_DEPTH);
    if (items.size > max_items && depth < Packed_BVH_Node::MAX_DEPTH) {
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
        u32           num_items = (u32)items.size;
        Sorting_Node *sorted_dims[6];
        ito(6) sorted_dims[i] = (Sorting_Node *)tl_alloc_tmp(sizeof(Sorting_Node) * num_items);
        ito(num_items) {
          float3 tmin, tmax;
          items[i].get_aabb(tmin, tmax);
          jto(3) {
            sorted_dims[j][i].val     = tmin[j];
            sorted_dims[j][i].id      = i;
            sorted_dims[j + 3][i].val = tmax[j];
            sorted_dims[j + 3][i].id  = i;
          }
        }
        ito(6)
            quicky_sort(sorted_dims[i], num_items,
                        [](Sorting_Node const &a, Sorting_Node const &b) { return a.val < b.val; });
        float max_dim_diff = 0.0f;
        u32   max_dim_id   = 0;
        u32   last_item    = num_items - 1;
        ito(3) {
          // max - min
          float diff = sorted_dims[i + 3][last_item].val - sorted_dims[i][0].val;
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
#endif

static float dot2(float3 a) { return dot(a, a); }

struct Triangle {
  float3 a;
  u32    surface_id;
  float3 b;
  u32    triangle_id;
  float3 c;
  u32    padding0;
  void   get_aabb(float3 &min, float3 &max) const {
    ito(3) min[i] = MIN(a[i], MIN(b[i], c[i]));
    ito(3) max[i] = MAX(a[i], MAX(b[i], c[i]));
  }
  AABB get_aabb() const {
    AABB out{};
    get_aabb(out.min, out.max);
    return out;
  }
  float2 get_end_points(u8 dim, float3 min, float3 max) const {
    float3 sp;
    ito(i) sp[i] = MIN(a[i], MIN(b[i], c[i]));
    float3 ep;
    ito(i) ep[i] = MAX(a[i], MAX(b[i], c[i]));

    bool fully_inside = //
        sp.x > min.x && //
        sp.y > min.y && //
        sp.z > min.z && //
        ep.x < max.x && //
        ep.y < max.y && //
        ep.z < max.z && //
        true;
    if (fully_inside) return float2{sp[dim], ep[dim]};
  }
  // https://www.iquilezles.org/www/articles/triangledistance/triangledistance.htm
  float distance(float3 p) {
    // prepare data
    float3 v21 = b - a;
    float3 p1  = p - a;
    float3 v32 = c - b;
    float3 p2  = p - b;
    float3 v13 = a - c;
    float3 p3  = p - c;
    float3 nor = cross(v21, v13);

    return -(dot(nor, p1) < 0.0f ? -1.0f : 1.0f) *
           sqrt( // inside/outside test
               (sign(dot(cross(v21, nor), p1)) + sign(dot(cross(v32, nor), p2)) +
                    sign(dot(cross(v13, nor), p3)) <
                2.0)
                   ?
                   // 3 edges
                   min(min(dot2(v21 * clamp(dot(v21, p1) / dot2(v21), 0.0f, 1.0f) - p1),
                           dot2(v32 * clamp(dot(v32, p2) / dot2(v32), 0.0f, 1.0f) - p2)),
                       dot2(v13 * clamp(dot(v13, p3) / dot2(v13), 0.0f, 1.0f) - p3))
                   :
                   // 1 face
                   dot(nor, p1) * dot(nor, p1) / dot2(nor));
  }
};
static_assert(sizeof(Triangle) == 48, "Blamey!");

#ifdef VULKII_EMBREE
namespace bvh {

struct Packed_BVH_Node {
  static constexpr f32 EPS       = 1.0e-3f;
  static constexpr u32 NULL_NODE = 0xffffffffu;
  float3               min;
  u32                  primitive_id;
  float3               max;
  u32                  next_on_miss;

  bool intersects(float3 tmin, float3 tmax) const {
    return                 //
        tmax.x >= min.x && //
        tmin.x <= max.x && //
        tmax.y >= min.y && //
        tmin.y <= max.y && //
        tmax.z >= min.z && //
        tmin.z <= max.z && //
        true;
  }
  bool intersects(float3 center, float radius) const {
    return                            //
        center.x + radius >= min.x && //
        center.x - radius <= max.x && //
        center.y + radius >= min.y && //
        center.y - radius <= max.y && //
        center.z + radius >= min.z && //
        center.z - radius <= max.z && //
        true;
  }
  bool inside(float3 tmin) const {
    return                 //
        tmin.x >= min.x && //
        tmin.x <= max.x && //
        tmin.y >= min.y && //
        tmin.y <= max.y && //
        tmin.z >= min.z && //
        tmin.z <= max.z && //
        true;
  }
  bool intersects_ray(float3 ro, float3 rd, float &min_t) const {
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
    min_t = nt;
    return true;
  }
  bool intersects_ray(float3 ro, float3 rd) const {
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
};
static_assert(sizeof(Packed_BVH_Node) == 32, "Blamey!");

// Based off
// https://github.com/kayru/RayTracedShadows/blob/master/Source/Shaders/RayTracedShadows.comp
struct BVH_Triangle {
  static constexpr f32 EPS = 1.0e-3f;

  float3 e0;
  u32    padding0;
  float3 e1;
  u32    next_on_miss;
  float3 v0;
  u32    next_on_hit;

  bool intersects_ray(float3 ro, float3 rd, float &min_t) const {
    const float3 s1   = cross(rd, e1);
    const float  invd = 1.0f / (dot(s1, e0));
    const float3 d    = ro.xyz - v0;
    const float  b1   = dot(d, s1) * invd;
    const float3 s2   = cross(d, e0);
    const float  b2   = dot(rd, s2) * invd;
    const float  temp = dot(e1, s2) * invd;

    if (b1 < 0.0f || b1 > 1.0f || b2 < 0.0f || b1 + b2 > 1.0f || temp < 0.0f || temp > min_t) {
      return false;
    } else {
      min_t = temp;
      return true;
    }
  }
};

static_assert(sizeof(BVH_Triangle) == 48, "Blamey!");

#  if 1 // Switching to embree
class Tri_BVH {
  AutoArray<float4> bvh_nodes{};
  Tri_BVH()  = default;
  ~Tri_BVH() = default;
  static Tri_BVH *create() {
    Tri_BVH *out = new Tri_BVH;
    return out;
  }
#    ifdef VULKII_EMBREE
  friend class Embree_BVH_Builder;
#    endif
  public:
  Packed_BVH_Node load_node(u32 index) {
    float4 *ptr = (float4 *)bvh_nodes.ptr + index * 2;
    return *(Packed_BVH_Node *)ptr;
  }
  BVH_Triangle load_triangle(u32 index) {
    float4 *     ptr          = bvh_nodes.ptr + index * 2;
    u32          primitive_id = *(reinterpret_cast<u32 *>(&ptr->w));
    float4       vertex       = bvh_nodes.ptr[primitive_id];
    BVH_Triangle tri          = *((BVH_Triangle *)ptr);
    tri.v0                    = vertex;
    return tri;
  }
  void                       release() { delete this; }
  template <typename F> void intersects_ray(float3 ro, float3 rd, F on_hit) {
    u32 cur_node_id = 0;
    // Closest hit on triangle
    f32 min_t    = 1.0e10f;
    u32 inter_id = Packed_BVH_Node::NULL_NODE;
    while (cur_node_id != Packed_BVH_Node::NULL_NODE) {
      Packed_BVH_Node cur_node = load_node(cur_node_id);
      if (cur_node.primitive_id != Packed_BVH_Node::NULL_NODE) { // Leaf Node
        BVH_Triangle tri{};
        tri.e0 = cur_node.min;
        tri.e1 = cur_node.max;
        tri.v0 = bvh_nodes.ptr[cur_node.primitive_id].xyz;
        if (tri.intersects_ray(ro, rd, min_t)) {
          inter_id = cur_node_id;
        }
      } else {
        float aabb_min_t = min_t;
        if (cur_node.intersects_ray(ro, rf, aabb_min_t)) {
          cur_node_id++;
        } else {
        }
      }
      cur_node_id = cur_node.next_on_miss;
    }
    return inter_id;
  }
};
#  endif
// Based off
// https://github.com/embree/embree/blob/master/tutorials/bvh_builder/bvh_builder_device.cpp
// https://interplayoflight.wordpress.com/2020/07/21/using-embree-generated-bvh-trees-for-gpu-raytracing/
struct Node {
  AABB          aabb{};
  virtual float sah() = 0;
  virtual ~Node() {}
  virtual Node *             get_child(u32 i) { return NULL; }
  virtual bool               is_leaf() { return false; }
  template <typename F> void traverse(F fn) {
    fn(this);
    if (get_child(0)) get_child(0)->traverse(fn);
    if (get_child(1)) get_child(1)->traverse(fn);
  }
};
struct InnerNode : public Node {
  Node *children[2]{};
  InnerNode() {}
  ~InnerNode() override {
    if (children[0]) delete children[0];
    if (children[1]) delete children[1];
  }
  Node *get_child(u32 i) override { return children[i]; }
  float sah() override {
    AABB &bounds0 = children[0]->aabb;
    AABB &bounds1 = children[1]->aabb;
    AABB  uni     = bounds0;
    uni.unite(bounds1);
    return 1.0f +
           (bounds0.area() * children[0]->sah() + bounds1.area() * children[1]->sah()) / uni.area();
  }
};
struct LeafNode : public Node {
  u32   surface_id  = 0;
  u32   triangle_id = 0;
  bool  is_leaf() override { return true; }
  float sah() override { return 1.0f; }
  ~LeafNode() override {}
  LeafNode(u32 surface_id, u32 triangle_id, const AABB &aabb)
      : surface_id(surface_id), triangle_id(triangle_id) {
    this->aabb = aabb;
  }
};
class Embree_BVH_Builder {
  public:
  private:
  RTCDevice device = NULL;
  // RTCScene  scene       = NULL;
  ~Embree_BVH_Builder() = default;
  Embree_BVH_Builder()  = default;

  static void *create_leaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive *prims,
                           size_t numPrims, void *userPtr) {
    assert(numPrims == 1);
    void *ptr = rtcThreadLocalAlloc(alloc, sizeof(LeafNode), 16);
    return (void *)new (ptr) LeafNode(prims->primID, prims->geomID, *(AABB *)prims);
  }

  static void *create_node(RTCThreadLocalAllocator alloc, unsigned int numChildren, void *userPtr) {
    assert(numChildren == 2);
    void *ptr = rtcThreadLocalAlloc(alloc, sizeof(InnerNode), 16);
    return (void *)new (ptr) InnerNode;
  }

  static void setChildren(void *nodePtr, void **childPtr, unsigned int numChildren, void *userPtr) {
    assert(numChildren == 2);
    for (size_t i = 0; i < 2; i++) ((InnerNode *)nodePtr)->children[i] = (Node *)childPtr[i];
  }

  static void setBounds(void *nodePtr, const RTCBounds **bounds, unsigned int numChildren,
                        void *userPtr) {
    assert(numChildren == 2);
    ((Node *)nodePtr)->aabb = *(const AABB *)bounds[0];
    ((Node *)nodePtr)->aabb.unite(*(const AABB *)bounds[1]);
  }
  static void splitPrimitive(const RTCBuildPrimitive *prim, unsigned int dim, float pos,
                             RTCBounds *lprim, RTCBounds *rprim, void *userPtr) {
    assert(dim < 3);
    // assert(prim->geomID == 0);
    *(RTCBuildPrimitive *)lprim = *(RTCBuildPrimitive *)prim;
    *(RTCBuildPrimitive *)rprim = *(RTCBuildPrimitive *)prim;
    (&lprim->upper_x)[dim]      = pos;
    (&rprim->lower_x)[dim]      = pos;
  }

  public:
  static Embree_BVH_Builder *create() {
    Embree_BVH_Builder *out = new Embree_BVH_Builder;
    out->device             = rtcNewDevice(NULL);
    return out;
  }
  void release() {
    rtcReleaseDevice(device);
    delete this;
  }
  struct BVH_Result {
    RTCBVH bvh  = NULL;
    Node * root = NULL;
    void   release() {
      if (bvh) {
        rtcReleaseBVH(bvh);
        // delete root;
      }
      MEMZERO(*this);
    }
  };
  BVH_Result build(Triangle *elems, size_t num_elems) {
    size_t             extra_nodes = num_elems;
    size_t             num_nodes   = num_elems;
    size_t             capacity    = num_elems + extra_nodes;
    RTCBuildPrimitive *prims       = tl_alloc_typed<RTCBuildPrimitive>(capacity);
    defer(tl_free(prims));
    ito(num_elems) {
      RTCBuildPrimitive prim{};
      AABB              aabb = elems[i].get_aabb();
      prim.lower_x           = aabb.min.x;
      prim.lower_y           = aabb.min.y;
      prim.lower_z           = aabb.min.z;
      prim.upper_x           = aabb.max.x;
      prim.upper_y           = aabb.max.y;
      prim.upper_z           = aabb.max.z;
      prim.geomID            = elems[i].triangle_id;
      prim.primID            = elems[i].surface_id;
      prims[i]               = prim;
    }

    RTCBVH bvh = rtcNewBVH(device);
    // defer(rtcReleaseBVH(bvh));
    RTCBuildArguments arguments      = rtcDefaultBuildArguments();
    arguments.byteSize               = sizeof(arguments);
    arguments.buildFlags             = RTC_BUILD_FLAG_NONE;
    arguments.buildQuality           = RTC_BUILD_QUALITY_HIGH;
    arguments.maxBranchingFactor     = 2;
    arguments.maxDepth               = 1024;
    arguments.sahBlockSize           = 1;
    arguments.minLeafSize            = 1;
    arguments.maxLeafSize            = 1;
    arguments.traversalCost          = 1.0f;
    arguments.intersectionCost       = 1.0f;
    arguments.bvh                    = bvh;
    arguments.primitives             = prims;
    arguments.primitiveCount         = num_nodes;
    arguments.primitiveArrayCapacity = capacity;
    arguments.createNode             = create_node;
    arguments.setNodeChildren        = setChildren;
    arguments.setNodeBounds          = setBounds;
    arguments.createLeaf             = create_leaf;
    arguments.splitPrimitive         = splitPrimitive;
    arguments.buildProgress          = nullptr;
    arguments.userPtr                = nullptr;
    BVH_Result out{};
    out.bvh  = bvh;
    out.root = (Node *)rtcBuildBVH(&arguments);
    return out;
  }
};
} // namespace bvh
#endif

// struct GPU_Meshlet {
//  u32 vertex_offset;
//  u32 index_offset;
//  u32 triangle_count;
//  u32 vertex_count;
//  // (x, y, z, radius)
//  float4 sphere;
//  // (x, y, z, _)
//  float4 cone_apex;
//  // (x, y, z, cutoff)
//  float4 cone_axis_cutoff;
//  u32    cone_pack;
//};
// static_assert(sizeof(GPU_Meshlet) == 68, "Packing error");

struct Ray {
  float3 o, d;
};

class Surface {
  public:
  Raw_Mesh_Opaque mesh;
  PBR_Material    material;

  void init() {
    mesh.init();
    material.init();
  }

  static Surface *create() {
    Surface *p = new Surface;
    p->init();
    return p;
  }

  void release() {
    mesh.release();
    delete this;
  }
};

class MeshNode : public Node {
  protected:
  Array<Surface *> surfaces;

  void init(string_ref name) {
    Node::init(name);
    surfaces.init();
  }

  public:
  DECLARE_TYPE(MeshNode, Node)

  static MeshNode *create(string_ref name) {
    MeshNode *out = new MeshNode;
    out->init(name);
    return out;
  }

  void     add_surface(Surface *surface) { surfaces.push(surface); }
  u32      getNumSurfaces() { return (u32)surfaces.size; }
  Surface *getSurface(u32 i) { return surfaces[i]; }
  Node *   clone() override {
    MeshNode *new_node = create(name.ref());
    Node::clone_into(new_node);
    return new_node;
  }
  void update() override {
    Node::update();
    if (surfaces.size)
      aabb = AABB{transform(surfaces[0]->mesh.min), transform(surfaces[0]->mesh.max)};
    else
      aabb = {};
    for (u32 i = 1; i < surfaces.size; i++) {
      aabb.unite(AABB{transform(surfaces[i]->mesh.min), transform(surfaces[i]->mesh.max)});
    }
  }
  void release() override { Node::release(); }
};

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
    f32 v_f32_min;
  };
  union {
    u32 v_u32_max;
    f32 v_f32_max;
  };
  bool               used = false;
  static Config_Item make_f32(f32 v, f32 min, f32 max) {
    Config_Item out;
    out.type      = F32;
    out.v_f32     = v;
    out.v_f32_min = min;
    out.v_f32_max = max;
    return out;
  }
  static Config_Item make_u32(u32 v, u32 min, u32 max) {
    Config_Item out;
    out.type      = U32;
    out.v_u32     = v;
    out.v_u32_min = min;
    out.v_u32_max = max;
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
            if (parse_decimal_int(l->get(1)->symbol.ptr, (u32)l->get(1)->symbol.len, &imin) ==
                false)
              parse_float(l->get(1)->symbol.ptr, (u32)l->get(1)->symbol.len, &fmin);
          } else if (l->cmp_symbol("max")) {
            if (parse_decimal_int(l->get(1)->symbol.ptr, (u32)l->get(1)->symbol.len, &imax) ==
                false)
              parse_float(l->get(1)->symbol.ptr, (u32)l->get(1)->symbol.len, &fmax);
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

  bool on_imgui() {
    AutoArray<string_t> to_remove{};
    bool                modified = false;
    AutoArray<string_t> sorted{};
    items.iter_pairs([&](string_t const &name, Config_Item &item) {
      if (item.used == false) return;
      sorted.push(name);
    });
    if (sorted.size)
      quicky_sort(&sorted[0], sorted.size, [&](string_t const &a, string_t const &b) {
        /*double av = 0.0;
        double bv = 0.0;
        ito(a.len()) av += pow(10.0, -1.0 * (i + 1)) * (double(a.buf[i]) / 128.0);
        ito(b.len()) bv += pow(10.0, -1.0 * (i + 1)) * (double(b.buf[i]) / 128.0);
        return av < bv;*/
        // if (a.len() < b.len()) return true;
        size_t len = min(a.len(), b.len());
        // for (i32 i = (i32)len - 1; i >= 0; i--) {
        for (i32 i = 0; i < (i32)len; i++) {
          if (a.buf[i] < b.buf[i]) return true;
        }
        return a.len() < b.len();
      });
    // items.iter_pairs([&](string_t const &name, Config_Item &item) {
    ito(sorted.size) {
      string_t const &name = sorted[i];
      Config_Item &   item = items.get_ref(name);
      if (item.used == false) continue;
      char buf[0x100];
      snprintf(buf, sizeof(buf), "%.*s", STRF(name.ref()));
      u64 hash = hash_of(name.ref());
      ImGui::PushID((i32)hash);
      defer(ImGui::PopID());
      if (item.type == Config_Item::U32) {
        if (item.v_u32_min != item.v_u32_max) {
          modified = ImGui::SliderInt(buf, (int *)&item.v_u32, item.v_u32_min, item.v_u32_max);
        } else {
          modified = ImGui::InputInt(buf, (int *)&item.v_u32);
        }
      } else if (item.type == Config_Item::F32) {
        if (item.v_f32_min != item.v_f32_max) {
          modified = ImGui::SliderFloat(buf, (float *)&item.v_f32, item.v_f32_min, item.v_f32_max);
          ImGui::SameLine();
          ImGui::SetNextItemWidth(20.0f);
          ImGui::InputFloat("min", (float *)&item.v_f32_min);
          ImGui::SameLine();
          ImGui::SetNextItemWidth(20.0f);
          ImGui::InputFloat("max", (float *)&item.v_f32_max);
        } else {
          modified = ImGui::InputFloat(buf, (float *)&item.v_f32);
        }
      } else if (item.type == Config_Item::BOOL) {
        modified = ImGui::Checkbox(buf, &item.v_bool);
      } else {
        TRAP;
      }
      ImGui::SameLine();
      if (ImGui::Button("Remove")) {
        to_remove.push(name);
      }
    }
    ito(to_remove.size) {
      modified = true;
      items.remove(to_remove[i]);
    }
    return modified;
  }

  u32 &get_u32(char const *name, u32 def = 0, u32 min = 0, u32 max = 10) {
    string_t _name;
    _name.init(stref_s(name));
    if (!items.contains(_name)) {
      items.insert(_name, Config_Item::make_u32(def, min, max));
    }
    ASSERT_DEBUG(items.contains(_name));
    auto &ref = items.get_ref(_name);
    ref.used  = true;
    return ref.v_u32;
  }

  f32 &get_f32(char const *name, f32 def = 0.0f, f32 min = -10.0f, f32 max = 10.0f) {
    string_t _name;
    _name.init(stref_s(name));
    if (!items.contains(_name)) {
      items.insert(_name, Config_Item::make_f32(def, min, max));
    }
    ASSERT_DEBUG(items.contains(_name));
    auto &ref = items.get_ref(_name);
    ref.used  = true;
    return ref.v_f32;
  }

  bool &get_bool(char const *name) {
    string_t _name;
    _name.init(stref_s(name));
    if (!items.contains(_name)) {
      items.insert(_name, {Config_Item::BOOL});
    }
    ASSERT_DEBUG(items.contains(_name));
    auto &ref = items.get_ref(_name);
    ref.type  = Config_Item::BOOL;
    ref.used  = true;
    return ref.v_bool;
  }

  void dump(FILE *file) {
    fprintf(file, "(config\n");
    items.iter_pairs([&](string_t const &name, Config_Item const &item) {
      if (item.type == Config_Item::U32) {
        if (item.v_u32_min != item.v_u32_max)
          fprintf(file, " (add u32 \"%.*s\" %i (min %i) (max %i))\n", STRF(name.ref()), item.v_u32,
                  item.v_u32_min, item.v_u32_max);
        else
          fprintf(file, " (add u32 \"%.*s\" %i)\n", STRF(name.ref()), item.v_u32);
      } else if (item.type == Config_Item::F32) {
        if (item.v_f32_min != item.v_f32_max)
          fprintf(file, " (add f32 \"%.*s\" %f (min %f) (max %f))\n", STRF(name.ref()), item.v_f32,
                  item.v_f32_min, item.v_f32_max);
        else
          fprintf(file, " (add f32 \"%.*s\" %f)\n", STRF(name.ref()), item.v_f32);
      } else if (item.type == Config_Item::BOOL) {
        fprintf(file, " (add bool \"%.*s\" %i)\n", STRF(name.ref()), item.v_bool ? 1 : 0);
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
    pos = float3(sinf(theta) * cosf(phi), cos(theta), sinf(theta) * sinf(phi)) * distance + look_at;
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
  Ray      gen_ray(float2 uv) {
    Ray r;
    r.o = pos;
    r.d = normalize(look + std::tan(fov * 0.5f) * (right * uv.x * aspect + up * uv.y));
    return r;
  }
};

class ISceneFactory {
  public:
  virtual Node *    add_node(string_ref name)                             = 0;
  virtual MeshNode *add_mesh_node(string_ref name)                        = 0;
  virtual Surface * add_surface(Raw_Mesh_Opaque &mesh, PBR_Material &mat) = 0;
  virtual u32       add_image(Image2D *img)                               = 0;
};

Node *               load_gltf_pbr(ISceneFactory *factory, string_ref filename);
Raw_Mesh_Opaque      optimize_mesh(Raw_Mesh_Opaque const &opaque_mesh);
Raw_Mesh_Opaque      simplify_mesh(Raw_Mesh_Opaque const &opaque_mesh);
Raw_Meshlets_Opaque *build_meshlets(Raw_Mesh_Opaque &opaque_mesh, size_t max_vertices = 64,
                                    size_t max_triangles = 124, float cone_weight = 0.5f);
void                 save_image(string_ref filename, Image2D const *image);
Image2D *            load_image(string_ref filename, rd::Format format = rd::Format::RGBA8_SRGBA);

class Asset_Manager {
  Array<Image2D *> images;
  Array<Surface *> surfaces;

  void init() { MEMZERO(*this); }

  public:
  Surface *add_surface(Raw_Mesh_Opaque &mesh, PBR_Material &mat) {
    Surface *p  = Surface::create();
    p->mesh     = mesh;
    p->material = mat;
    surfaces.push(p);
    return p;
  }
  static Asset_Manager *create() {
    Asset_Manager *out = new Asset_Manager;
    out->init();
    return out;
  }
  u32 add_image(Image2D *img) {
    images.push(img);
    return (u32)images.size - 1;
  }
  u32 load_image(string_ref path, rd::Format format) {
    images.push(::load_image(path, format));
    return (u32)images.size - 1;
  }
  Image2D const *get_image(u32 index) { return images[index]; }
  void           release() {
    ito(surfaces.size) surfaces[i]->release();
    ito(images.size) images[i]->release();
    images.release();
    surfaces.release();
    delete this;
  }
  Array<Image2D *> const &get_images() const { return images; }
  Array<Surface *> const &get_surfaces() const { return surfaces; }
};

class Scene {
  public:
  Node *         root;
  Asset_Manager *assets;

  friend class SceneFactory;
  class SceneFactory : public ISceneFactory {
public:
    Scene *scene;
    SceneFactory(Scene *scene) : scene(scene) {}
    Node *    add_node(string_ref name) override { return Node::create(name); }
    MeshNode *add_mesh_node(string_ref name) override { return MeshNode::create(name); }
    Surface * add_surface(Raw_Mesh_Opaque &mesh, PBR_Material &mat) override {
      return scene->assets->add_surface(mesh, mat);
    }
    u32 add_image(Image2D *img) override { return scene->assets->add_image(img); }
  };

  void load_mesh(string_ref name, string_ref path) {
    SceneFactory sf(this);
    root->add_child(load_gltf_pbr(&sf, path)->rename(name));
  }

  template <typename F> void traverse(F fn, Node *node) {
    if (node == NULL) node = root;
    fn(node);
    ito(node->get_children().size) { traverse(fn, node->get_children()[i]); }
  }
  void init() {
    MEMZERO(*this);
    root   = Node::create(stref_s("ROOT"));
    assets = Asset_Manager::create();
  }

  public:
  static Scene *create() {
    Scene *s = new Scene;
    s->init();
    return s;
  }

  void restore(List *l) {
    if (l == NULL) return;
    if (l->child) {
      restore(l->child);
      restore(l->next);
    } else {
      if (l->symbol.eq("scene")) {
        root->restore(l->next);
      }
    }
    // if (l == NULL) return;
    // if (l->child) {
    // restore(l);
    // restore(l->next);
    //} else {
    // if (l->cmp_symbol("node")) {
    //}
    //}
  }
  void save(String_Builder &sb) {
    sb.putf("(scene\n");
    root->save(sb);
    sb.putf(")\n");
  }

  Asset_Manager *get_assets() { return assets; }
  Node *         get_root() { return root; }
  Node *         get_node(string_ref name) {
    Node *out = NULL;
    traverse([&](Node *node) {
      if (node->get_name() == name) {
        out = node;
      }
    });
    return out;
  }
  template <typename F> void traverse(F fn) { traverse(fn, root); }
  void                       update() { root->update(); }
  void                       release() {
    root->release();
    assets->release();
    delete this;
  }
};

class GfxSurface {
  rd::IDevice *factory;
  Surface *    surface;

  public:
  InlineArray<size_t, 16>    attribute_offsets{};
  InlineArray<size_t, 16>    attribute_sizes{};
  InlineArray<Attribute, 16> attributes{};

  size_t total_memory_needed = 0;
  u32    total_indices       = 0;
  size_t index_offset        = 0;

  Resource_ID buffer{};
  rd::Index_t index_type{};

  private:
  void init(rd::IDevice *factory, Surface *surface) {
    MEMZERO(*this);
    this->index_type    = surface->mesh.index_type;
    this->factory       = factory;
    this->surface       = surface;
    total_memory_needed = 0;
    total_indices       = 0;
    index_offset        = 0;
    attribute_offsets.init();
    attribute_sizes.init();
    attributes.init();

    rd::Buffer_Create_Info info;
    MEMZERO(info);
    info.memory_type = rd::Memory_Type::GPU_LOCAL;
    info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER |
                      (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER |
                      (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                      (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
    info.size = (u32)get_needed_memory();
    buffer    = factory->create_buffer(info);

    MEMZERO(info);
    info.memory_type          = rd::Memory_Type::CPU_WRITE_GPU_READ;
    info.usage_bits           = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
    info.size                 = (u32)get_needed_memory();
    Resource_ID stagin_buffer = factory->create_buffer(info);
    defer(factory->release_resource(stagin_buffer));

    InlineArray<size_t, 16> attribute_cursors;
    MEMZERO(attribute_cursors);
    size_t indices_offset = 0;
    void * ptr            = factory->map_buffer(stagin_buffer);
    jto(surface->mesh.attributes.size) {
      Attribute attribute      = surface->mesh.attributes[j];
      size_t    attribute_size = surface->mesh.get_attribute_size(j);
      memcpy((u8 *)ptr + attribute_offsets[j] + attribute_cursors[j],
             &surface->mesh.attribute_data[0] + attribute.offset, attribute_size);
      attribute_cursors[j] += attribute_size;
    }
    size_t index_size = surface->mesh.get_bytes_per_index() * surface->mesh.num_indices;
    memcpy((u8 *)ptr + index_offset + indices_offset, &surface->mesh.index_data[0], index_size);
    indices_offset += index_size;
    factory->unmap_buffer(stagin_buffer);
    auto *ctx = factory->start_compute_pass();
    ctx->copy_buffer(stagin_buffer, 0, buffer, 0, (u32)get_needed_memory());
    factory->end_compute_pass(ctx);
    total_indices += surface->mesh.num_indices;
  }
  size_t get_needed_memory() {
    if (total_memory_needed == 0) {
      ito(surface->mesh.attributes.size) {
        attributes.push(surface->mesh.attributes[i]);
        attribute_offsets.push(0);
        attribute_sizes.push(0);
      }
      jto(surface->mesh.attributes.size) {
        attribute_sizes[j] += surface->mesh.get_attribute_size(j);
      }

      ito(attributes.size) {
        jto(i) { attribute_offsets[i] += attribute_sizes[j]; }
        attribute_offsets[i] = rd::IDevice::align_up(attribute_offsets[i]);
        total_memory_needed  = attribute_offsets[i] + attribute_sizes[i];
      }
      total_memory_needed =
          rd::IDevice::align_up(total_memory_needed, rd::IDevice::BUFFER_ALIGNMENT);
      index_offset = total_memory_needed;
      total_memory_needed += surface->mesh.get_bytes_per_index() * surface->mesh.num_indices;
    }
    return total_memory_needed;
  }

  u32 get_num_indices() { return total_indices; }

  public:
  size_t get_bytes_per_index() { return surface->mesh.get_bytes_per_index(); }
  size_t get_attribute_offset(rd::Attriute_t type) {
    ito(attributes.size) {
      if (attributes[i].type == type) {
        return attribute_offsets[i];
      }
    }
    TRAP;
  }
  size_t get_attribute_stride(rd::Attriute_t type) {
    ito(attributes.size) {
      if (attributes[i].type == type) {
        return attributes[i].stride;
      }
    }
    TRAP;
  }
  static GfxSurface *create(rd::IDevice *factory, Surface *surface) {
    GfxSurface *out = new GfxSurface;
    out->init(factory, surface);
    return out;
  }
  void release() {
    factory->release_resource(buffer);
    delete this;
  }
  void setup_attributes(rd::Graphics_Pipeline_State &state, u32 *attribute_to_location) {
    ito(attributes.size) {
      Attribute attr = attributes[i];
      state.IA_set_vertex_binding(i, attr.stride, rd::Input_Rate::VERTEX);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = i;
      info.format   = attr.format;
      info.location = attribute_to_location[(u32)attr.type];
      info.offset   = 0;
      info.type     = attr.type;
      state.IA_set_attribute(info);
    }
  }

  void draw(rd::ICtx *ctx, rd::Graphics_Pipeline_State const &gfx_state) {
    jto(gfx_state.num_attributes) {
      ito(attributes.size) {
        if (gfx_state.attributes[j].type == attributes[i].type) {
          ctx->bind_vertex_buffer(gfx_state.attributes[j].binding, buffer, attribute_offsets[i]);
        }
      }
    }
    ctx->bind_index_buffer(buffer, index_offset, index_type);
    u32 vertex_cursor = 0;
    u32 index_cursor  = 0;
    ctx->draw_indexed(surface->mesh.num_indices, 1, index_cursor, 0, vertex_cursor);
    index_cursor += surface->mesh.num_indices;
    vertex_cursor += surface->mesh.num_vertices;
  }
};

class GfxSufraceComponent : public Node::Component {
  Array<GfxSurface *> gfx_surfaces{};

  public:
  DECLARE_TYPE(GfxSufraceComponent, Component)

  ~GfxSufraceComponent() override {}
  GfxSufraceComponent(Node *n) : Component(n) {}
  static GfxSufraceComponent *create(rd::IDevice *factory, Node *n) {
    ASSERT_DEBUG(n->isa<MeshNode>());
    GfxSufraceComponent *s  = new GfxSufraceComponent(n);
    MeshNode *           mn = n->dyn_cast<MeshNode>();
    ito(mn->getNumSurfaces()) {
      s->gfx_surfaces.push(GfxSurface::create(factory, mn->getSurface(i)));
    }
    n->addComponent(s);
    return s;
  }
  u32         getNumSurfaces() { return (u32)gfx_surfaces.size; }
  GfxSurface *getSurface(u32 i) { return gfx_surfaces[i]; }
  void        release() override {
    ito(gfx_surfaces.size) gfx_surfaces[i]->release();
    gfx_surfaces.release();
    Component::release();
  }
};

class BVHSufraceComponent : public Node::Component {
#ifdef VULKII_EMBREE
  bvh::Embree_BVH_Builder::BVH_Result bvh{};
#endif
  public:
  DECLARE_TYPE(BVHSufraceComponent, Component)
#ifdef VULKII_EMBREE
  bvh::Node *getBVH() { return bvh.root; }
#else
  bvh::Node *getBVH() { return NULL; }
#endif

  ~BVHSufraceComponent() override {}
  BVHSufraceComponent(Node *n) : Component(n) {}
  static BVHSufraceComponent *create(Node *n) {
    ASSERT_DEBUG(n->isa<MeshNode>());
    BVHSufraceComponent *s  = new BVHSufraceComponent(n);
    MeshNode *           mn = n->dyn_cast<MeshNode>();
    n->addComponent(s);
    s->updateBVH();
    return s;
  }
  void updateBVH() {
#ifdef VULKII_EMBREE
    if (bvh.root) bvh.release();
    static bvh::Embree_BVH_Builder *bvh_builder = bvh::Embree_BVH_Builder::create();
    AutoArray<Triangle>             triangles{};
    MeshNode *                      mn = node->dyn_cast<MeshNode>();
    ASSERT_DEBUG(mn);
    ito(mn->getNumSurfaces()) {
      Surface *s = mn->getSurface(i);
      triangles.reserve(s->mesh.num_indices / 3);
      jto(s->mesh.num_indices / 3) {
        Triangle_Full ftri = s->mesh.fetch_triangle(j);
        Triangle      t{};
        t.a           = mn->transform(ftri.v0.position);
        t.b           = mn->transform(ftri.v1.position);
        t.c           = mn->transform(ftri.v2.position);
        t.surface_id  = i;
        t.triangle_id = j;
        triangles.push(t);
      }
    }
    bvh = bvh_builder->build(&triangles[0], triangles.size);
#endif
  }
  void release() override {
#ifdef VULKII_EMBREE
    bvh.release();
#endif
    Component::release();
  }
};

#if 1

class GfxMeshletSurface {
  rd::IDevice *        factory  = NULL;
  Raw_Meshlets_Opaque *meshlets = NULL;

  public:
  InlineArray<size_t, 16>    attribute_offsets{};
  InlineArray<size_t, 16>    attribute_sizes{};
  InlineArray<Attribute, 16> attributes{};

  size_t total_memory_needed = 0;
  u32    total_indices       = 0;
  size_t index_offset        = 0;

  Resource_ID buffer{};

  private:
  void init(rd::IDevice *factory, Raw_Meshlets_Opaque *meshlets) {
    MEMZERO(*this);
    this->factory       = factory;
    this->meshlets      = meshlets;
    total_memory_needed = 0;
    total_indices       = 0;
    index_offset        = 0;
    attribute_offsets.init();
    attribute_sizes.init();
    attributes.init();

    rd::Buffer_Create_Info info;
    MEMZERO(info);
    info.memory_type = rd::Memory_Type::GPU_LOCAL;
    info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER |
                      (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER |
                      (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                      (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
    info.size = (u32)get_needed_memory();
    buffer    = factory->create_buffer(info);

    MEMZERO(info);
    info.memory_type          = rd::Memory_Type::CPU_WRITE_GPU_READ;
    info.usage_bits           = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
    info.size                 = (u32)get_needed_memory();
    Resource_ID stagin_buffer = factory->create_buffer(info);
    defer(factory->release_resource(stagin_buffer));

    InlineArray<size_t, 16> attribute_cursors;
    MEMZERO(attribute_cursors);
    size_t indices_offset = 0;
    void * ptr            = factory->map_buffer(stagin_buffer);
    jto(meshlets->attributes.size) {
      Attribute attribute      = meshlets->attributes[j];
      size_t    attribute_size = meshlets->get_attribute_size(j);
      memcpy((u8 *)ptr + attribute_offsets[j] + attribute_cursors[j],
             &meshlets->attribute_data[0] + attribute.offset, attribute_size);
      attribute_cursors[j] += attribute_size;
    }
    size_t index_size = sizeof(u8) * meshlets->index_data.size;
    memcpy((u8 *)ptr + index_offset + indices_offset, &meshlets->index_data[0], index_size);
    indices_offset += index_size;
    factory->unmap_buffer(stagin_buffer);
    auto *ctx = factory->start_compute_pass();
    ctx->copy_buffer(stagin_buffer, 0, buffer, 0, (u32)get_needed_memory());
    factory->end_compute_pass(ctx);
    total_indices += meshlets->index_data.size;
  }
  size_t get_needed_memory() {
    if (total_memory_needed == 0) {
      ito(meshlets->attributes.size) {
        attributes.push(meshlets->attributes[i]);
        attribute_offsets.push(0);
        attribute_sizes.push(0);
      }
      jto(meshlets->attributes.size) { attribute_sizes[j] += meshlets->get_attribute_size(j); }

      ito(attributes.size) {
        jto(i) { attribute_offsets[i] += attribute_sizes[j]; }
        attribute_offsets[i] = rd::IDevice::align_up(attribute_offsets[i]);
        total_memory_needed  = attribute_offsets[i] + attribute_sizes[i];
      }
      total_memory_needed =
          rd::IDevice::align_up(total_memory_needed, rd::IDevice::BUFFER_ALIGNMENT);
      index_offset = total_memory_needed;
      total_memory_needed += sizeof(u8) * meshlets->index_data.size;
    }
    return total_memory_needed;
  }

  u32 get_num_indices() { return total_indices; }

  public:
  size_t get_attribute_offset(rd::Attriute_t type) {
    ito(attributes.size) {
      if (attributes[i].type == type) {
        return attribute_offsets[i];
      }
    }
    TRAP;
  }
  size_t get_attribute_stride(rd::Attriute_t type) {
    ito(attributes.size) {
      if (attributes[i].type == type) {
        return attributes[i].stride;
      }
    }
    TRAP;
  }
  size_t                    get_index_offset() { return index_offset; }
  static GfxMeshletSurface *create(rd::IDevice *factory, Raw_Meshlets_Opaque *meshlets) {
    GfxMeshletSurface *out = new GfxMeshletSurface;
    out->init(factory, meshlets);
    return out;
  }
  void release() {
    factory->release_resource(buffer);
    delete this;
  }
  void setup_attributes(rd::Graphics_Pipeline_State &state, u32 *attribute_to_location) {
    ito(attributes.size) {
      Attribute attr = attributes[i];
      state.IA_set_vertex_binding(i, attr.stride, rd::Input_Rate::VERTEX);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = i;
      info.format   = attr.format;
      info.location = attribute_to_location[(u32)attr.type];
      info.offset   = 0;
      info.type     = attr.type;
      state.IA_set_attribute(info);
    }
  }
  void bind_uav_buffer(rd::IBinding_Table *table, u32 set, u32 binding) {
    table->bind_UAV_buffer(set, binding, buffer, 0, 0);
  }
  template <typename F> void iterate(F f) {
    /* jto(gfx_state.num_attributes) {
       ito(attributes.size) {
         if (gfx_state.attributes[j].type == attributes[i].type) {
           ctx->bind_vertex_buffer(gfx_state.attributes[j].binding, buffer, attribute_offsets[i]);
         }
       }
     }
     ctx->bind_index_buffer(buffer, index_offset, index_type);*/
    ito(meshlets->meshlets.size) { f(meshlets->meshlets[i]); }
    // u32 vertex_cursor = 0;
    // u32 index_cursor  = 0;
    // ctx->draw_indexed(surface->mesh.num_indices, 1, index_cursor, 0, vertex_cursor);
    // index_cursor += surface->mesh.num_indices;
    // vertex_cursor += surface->mesh.num_vertices;
  }
};
#endif // 0

class MeshletSufraceComponent : public Node::Component {
  Array<Raw_Meshlets_Opaque *> meshlets{};

  public:
  DECLARE_TYPE(MeshletSufraceComponent, Component)

  ~MeshletSufraceComponent() override {}
  MeshletSufraceComponent(Node *n) : Component(n) {}
  static MeshletSufraceComponent *create(Node *n, size_t max_vertices = 64,
                                         size_t max_triangles = 124, float cone_weight = 0.5f) {
    ASSERT_DEBUG(n->isa<MeshNode>());
    MeshletSufraceComponent *s  = new MeshletSufraceComponent(n);
    MeshNode *               mn = n->dyn_cast<MeshNode>();
    ito(mn->getNumSurfaces()) {
      Surface *sf       = mn->getSurface(i);
      auto     meshlets = build_meshlets(sf->mesh, max_vertices, max_triangles, cone_weight);
      s->meshlets.push(meshlets);
    }
    n->addComponent(s);
    return s;
  }
  Raw_Meshlets_Opaque *get_meshlets(u32 i) {
    if (i >= meshlets.size) return NULL;
    return meshlets[i];
  }

  void release() override {
    ito(meshlets.size) meshlets[i]->release();
    meshlets.release();
    Component::release();
  }
};

class GfxMeshletSufraceComponent : public Node::Component {
  Array<GfxMeshletSurface *> gfx_meshlets{};

  public:
  DECLARE_TYPE(GfxMeshletSufraceComponent, Component)

  ~GfxMeshletSufraceComponent() override {}
  GfxMeshletSufraceComponent(Node *n) : Component(n) {}
  static GfxMeshletSufraceComponent *create(rd::IDevice *factory, Node *n) {
    ASSERT_DEBUG(n->isa<MeshNode>());
    GfxMeshletSufraceComponent *s   = new GfxMeshletSufraceComponent(n);
    MeshNode *                  mn  = n->dyn_cast<MeshNode>();
    MeshletSufraceComponent *   msc = n->getComponent<MeshletSufraceComponent>();
    ASSERT_DEBUG(msc);
    ito(mn->getNumSurfaces()) {
      Surface *sf       = mn->getSurface(i);
      auto     meshlets = msc->get_meshlets(i);
      s->gfx_meshlets.push(GfxMeshletSurface::create(factory, meshlets));
    }
    n->addComponent(s);
    return s;
  }
  GfxMeshletSurface *get_meshlets(u32 i) {
    if (i >= gfx_meshlets.size) return NULL;
    return gfx_meshlets[i];
  }

  void release() override {
    ito(gfx_meshlets.size) gfx_meshlets[i]->release();
    gfx_meshlets.release();
    Component::release();
  }
};

class Topo_Mesh {
  private:
  Topo_Mesh()  = default;
  ~Topo_Mesh() = default;

  public:
  static Topo_Mesh *create() { return new Topo_Mesh; }
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
    u32  origin;
    u32  end;
    u32  face;
    i32  sibling;
    u32  next_edge;
    u32  prev_edge;
    bool is_seam;

    void init() { memset(this, 0, sizeof(Edge)); }
    void release() {}
  };
  struct Vertex {
    SmallArray<u32, 8> edges;
    SmallArray<u32, 8> faces;
    u32                index;
    float3             pos;
    union {
      float3 f3param0;
      float4 f4param0;
      bool   bparam0;
    };
    union {
      float3 f3param1;
      float4 f4param1;
      bool   bparam1;
    };
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
    return (u32)edges.size - 1;
  }
  TriFace *get_face(u32 id) { return &faces[id]; }
  u32      add_face() {
    faces.push({});
    return (u32)faces.size - 1;
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
        {
          e->is_seam = true;
          seam_edges.push(i);
        }
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
    /* quicky_sort(&seam_edges[0], seam_edges.size,
                 [&](u32 se0, u32 se1) { return edges[se0].origin < edges[se1].origin; });*/
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
    delete this;
  }
};

class TopomeshComponent : public Node::Component {
  AutoArray<Topo_Mesh *> topo_meshes{};

  public:
  DECLARE_TYPE(TopomeshComponent, Component)

  ~TopomeshComponent() override {}
  TopomeshComponent(Node *n) : Component(n) {}
  static TopomeshComponent *create(Node *n) {
    ASSERT_DEBUG(n->isa<MeshNode>());
    TopomeshComponent *s  = new TopomeshComponent(n);
    MeshNode *         mn = n->dyn_cast<MeshNode>();
    ito(mn->getNumSurfaces()) {
      auto topo_mesh = Topo_Mesh::create();
      topo_mesh->init(mn->getSurface(i)->mesh);
      s->topo_meshes.push(topo_mesh);
    }

    n->addComponent(s);
    return s;
  }
  u32        get_num_meshes() { return topo_meshes.size; }
  Topo_Mesh *get_topo(u32 i) { return topo_meshes[i]; }
  void       release() override {
    ito(topo_meshes.size) { topo_meshes[i]->release(); }
    Component::release();
  }
};

#endif // SCENE