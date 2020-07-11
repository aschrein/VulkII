#include "rt.hpp"

//#define TRACY_ENABLE 1
//#define TRACY_HAS_CALLSTACK 1
//#define TRACY_NO_EXIT 1
#include <tracy/Tracy.hpp>

#define UTILS_TL_IMPL 1
//#define UTILS_TL_IMPL_DEBUG
//#define UTILS_TL_IMPL_TRACY 1
#define UTILS_TL_TMP_SIZE 1 << 27
#include "utils.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#define PACK_SIZE 16

//#ifdef UTILS_AVX512

#include <immintrin.h>

struct vbool {
  static constexpr u32 WIDTH = 16;
  __mmask16            val;
  bool                 any() { return val != 0u; }
  bool                 all() { return val == 0xffffu; }
  bool                 none() { return val == 0u; }

  operator __mmask16() { return val; }

  bool        is_enabled(u32 i) const { return (val & (1 << i)) != 0; }
  void        enable(u32 i) { val = val | (1 << i); }
  void        disable(u32 i) { val = val & ~(1 << i); }
  u32         popcnt() { return __popcnt16(val); }
  inline bool lsb(u32 &out) {
    unsigned long ret;
    char          c = _BitScanForward(&ret, (long)val);
    out             = (u32)ret;
    return c != 0;
  }
  inline bool take_lsb(u32 &out) {
    if (lsb(out)) {
      disable(out);
      return true;
    }
    return false;
  }
  vbool operator!() const {
    vbool b;
    b.val = ~val;
    return b;
  }
  vbool operator~() const {
    vbool b;
    b.val = ~val;
    return b;
  }
  vbool operator&&(vbool const &that) const {
    vbool b;
    b.val = val & that.val;
    return b;
  }
  vbool operator&(i32 that) const {
    vbool b;
    b.val = val & (u16)that;
    return b;
  }
  vbool operator&(u32 that) const {
    vbool b;
    b.val = val & (u16)that;
    return b;
  }
  vbool operator&(u16 that) const {
    vbool b;
    b.val = val & (u16)that;
    return b;
  }
  vbool operator|(i32 that) const {
    vbool b;
    b.val = val | (u16)that;
    return b;
  }
  vbool operator|(u32 that) const {
    vbool b;
    b.val = val | (u16)that;
    return b;
  }
  vbool operator|(u16 that) const {
    vbool b;
    b.val = val | (u16)that;
    return b;
  }
  vbool operator&(vbool const &that) const {
    vbool b;
    b.val = val & that.val;
    return b;
  }
  vbool operator||(vbool const &that) const {
    vbool b;
    b.val = val | that.val;
    return b;
  }
  vbool operator|(vbool const &that) const {
    vbool b;
    b.val = val | that.val;
    return b;
  }
  void set(u32 i) { val |= (1 << i); }
  void set(u32 i, bool t) { val = (val & ~(1 << i)) | ((t ? 1 : 0) << i); }
  void dump() const {
    fprintf(stdout, "vbool: ");
    ito(16) fprintf(stdout, "%i", is_enabled(i) ? 1 : 0);
    fprintf(stdout, "\n");
  }
};

struct TL_Mask {
  vbool  current;
  vbool  stack[0x10];
  size_t stack_cursor = 0;
  void   push() {
    stack[stack_cursor++] = current;
    ASSERT_DEBUG(stack_cursor <= ARRAY_SIZE(stack));
  }
  void pop() {
    ASSERT_DEBUG(stack_cursor > 0);
    current = stack[--stack_cursor];
  }
  void      set(vbool b) { current = b; }
  vbool &   cur() { return current; }
  __mmask16 get() { return current.val; }
  void      enable_all() { current.val = ~0u; }
  void      disable(u32 index) { current.val &= ~(1 << index); }
  void      dump() {
    fprintf(stdout, "MASK: ");
    ito(16) { fprintf(stdout, "%i", ((current.val & (1 << i)) != 0) ? 1 : 0); }
    fprintf(stdout, "\n");
  }
};

thread_local TL_Mask tl_mask;

static inline TL_Mask &mask() { return tl_mask; }
static inline vbool &  cur_vmask() { return tl_mask.current; }

#define VIF(cond)                                                              \
  do {                                                                         \
    mask().push();                                                             \
    cur_vmask() = cond;                                                        \
  } while (0)
#define VENDIF()                                                               \
  do {                                                                         \
    mask().pop();                                                              \
  } while (0)

struct vint {
  static constexpr u32 WIDTH = 16;
  union {
    __m512i val;
    i32     raw[16];
  };
  vint operator+(vint const &that) const {
    vint r;
    r.val = _mm512_maskz_add_epi32(cur_vmask(), this->val, that.val);
    return r;
  }
  vint operator-(vint const &that) const {
    vint r;
    r.val = _mm512_maskz_sub_epi32(cur_vmask(), this->val, that.val);
    return r;
  }
  vint operator*(vint const &that) const {
    vint r;
    r.val = _mm512_maskz_mul_epi32(cur_vmask(), this->val, that.val);
    return r;
  }
  static vint splat(i32 a) {
    vint r;
    r.val = __m512i{};
    return r;
  }
  vbool operator<(vint const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmplt_epi32_mask(this->val, that.val);
    return b;
  }
  vbool operator<=(vint const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmple_epi32_mask(this->val, that.val);
    return b;
  }
  vbool operator>(vint const &that) const {
    return vbool{cur_vmask()} & !(*this <= that);
  }
  vbool operator>=(vint const &that) const {
    return vbool{cur_vmask()} & !(*this < that);
  }
  vbool operator==(vint const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmpeq_epi32_mask(this->val, that.val);
    return b;
  }
  vbool operator!=(vint const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmpneq_epi32_mask(this->val, that.val);
    return b;
  }
  i32 &operator[](u32 i) { return raw[i]; }
  i32  operator[](u32 i) const { return raw[i]; }
  void dump() const {
    fprintf(stdout, "vint:\n");
    ito(16) fprintf(stdout, " %i ", (*this)[i]);
    fprintf(stdout, "\n");
  }
};

#define SOP(OP)                                                                \
  static inline vint operator OP(i32 a, vint const &vf) {                      \
    vint         va = vint::splat(a);                                          \
    return va OP vf;                                                           \
  }
SOP(*)
SOP(+)
SOP(-)
#undef SOP

#define SOP(OP)                                                                \
  static inline vint operator OP(vint const &vf, i32 a) {                      \
    vint         va = vint::splat(a);                                          \
    return vf OP va;                                                           \
  }
SOP(*)
SOP(+)
SOP(-)
#undef SOP

#define SOP(OP)                                                                \
  static inline vbool operator OP(i32 a, vint const &vf) {                     \
    vint         va = vint::splat(a);                                          \
    return va OP vf;                                                           \
  }
SOP(<)
SOP(<=)
SOP(>)
SOP(>=)
SOP(==)
SOP(!=)
#undef SOP

#define SOP(OP)                                                                \
  static inline vbool operator OP(vint const &vf, i32 a) {                     \
    vint         va = vint::splat(a);                                          \
    return vf OP va;                                                           \
  }
SOP(<)
SOP(<=)
SOP(>)
SOP(>=)
SOP(==)
SOP(!=)
#undef SOP

struct vfloat {
  static constexpr u32 WIDTH = 16;
  union {
    __m512 val;
    f32    raw[16];
  };
  vfloat operator+(vfloat const &that) const {
    vfloat r;
    r.val = _mm512_maskz_add_ps(cur_vmask(), this->val, that.val);
    return r;
  }
  vfloat operator-(vfloat const &that) const {
    vfloat r;
    r.val = _mm512_maskz_sub_ps(cur_vmask(), this->val, that.val);
    return r;
  }
  vfloat operator/(vfloat const &that) const {
    vfloat r;
    r.val = _mm512_maskz_div_ps(cur_vmask(), this->val, that.val);
    return r;
  }
  vfloat operator*(vfloat const &that) const {
    vfloat r;
    r.val = _mm512_maskz_mul_ps(cur_vmask(), this->val, that.val);
    return r;
  }
  vfloat        operator-() const { return *this * splat(-1.0f); }
  static vfloat splat(f32 a) {
    vfloat r;
    r.val = _mm512_broadcastss_ps(_mm_set_ss(a));
    return r;
  }
  vbool operator<(vfloat const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmplt_ps_mask(this->val, that.val);
    return b;
  }
  vbool operator<=(vfloat const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmple_ps_mask(this->val, that.val);
    return b;
  }
  vbool operator>(vfloat const &that) const {
    return cur_vmask() && !(*this <= that);
  }
  vbool operator>=(vfloat const &that) const {
    return cur_vmask() && !(*this < that);
  }
  vbool operator==(vfloat const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmpeq_ps_mask(this->val, that.val);
    return b;
  }
  vbool operator!=(vfloat const &that) const {
    vbool b;
    b.val = cur_vmask() & _mm512_cmpneq_ps_mask(this->val, that.val);
    return b;
  }
  static vfloat blend(vfloat const &a, vfloat const &b, vbool const &k) {
    vfloat r;
    r.val = _mm512_mask_mov_ps(a.val, k.val, b.val);
    return r;
  }
  f32 &operator[](u32 i) { return raw[i]; }
  f32  operator[](u32 i) const { return raw[i]; }
  void dump() const {
    fprintf(stdout, "vfloat:\n");
    ito(16) fprintf(stdout, " %f ", (*this)[i]);
    fprintf(stdout, "\n");
  }
#define VOP(OP)                                                                \
  vfloat operator OP##=(vfloat const &that) {                                  \
    *this = *this OP that;                                                     \
    return *this;                                                              \
  }
  VOP(+)
  VOP(-)
  VOP(*)
  VOP(/)
#undef VOP
#define VOP(OP)                                                                \
  vfloat operator OP##=(f32 that) {                                            \
    *this = *this OP splat(that);                                              \
    return *this;                                                              \
  }
  VOP(+)
  VOP(-)
  VOP(*)
  VOP(/)
#undef VOP
};

#define SOP(OP)                                                                \
  static inline vfloat operator OP(f32 a, vfloat const &vf) {                  \
    vfloat       va = vfloat::splat(a);                                        \
    return va OP vf;                                                           \
  }
SOP(/)
SOP(*)
SOP(+)
SOP(-)
#undef SOP

#define SOP(OP)                                                                \
  static inline vfloat operator OP(vfloat const &vf, f32 a) {                  \
    vfloat       va = vfloat::splat(a);                                        \
    return vf OP va;                                                           \
  }
SOP(/)
SOP(*)
SOP(+)
SOP(-)
#undef SOP

#define SOP(OP)                                                                \
  static inline vbool operator OP(f32 a, vfloat const &vf) {                   \
    vfloat       va = vfloat::splat(a);                                        \
    return va OP vf;                                                           \
  }
SOP(<)
SOP(<=)
SOP(>)
SOP(>=)
SOP(==)
SOP(!=)
#undef SOP

#define SOP(OP)                                                                \
  static inline vbool operator OP(vfloat const &vf, f32 a) {                   \
    vfloat       va = vfloat::splat(a);                                        \
    return vf OP va;                                                           \
  }
SOP(<)
SOP(<=)
SOP(>)
SOP(>=)
SOP(==)
SOP(!=)
#undef SOP

struct vfloat3 {
  static constexpr u32 WIDTH = 16;
  vfloat               x;
  vfloat               y;
  vfloat               z;
#define VOP(OP)                                                                \
  vfloat3 operator OP(vfloat3 const &that) const {                             \
    vfloat3          r;                                                        \
    r.x = this->x OP that.x;                                                   \
    r.y = this->y OP that.y;                                                   \
    r.z = this->z OP that.z;                                                   \
    return r;                                                                  \
  }
  VOP(+)
  VOP(-)
  VOP(*)
  VOP(/)
#undef VOP
#define VOP(OP)                                                                \
  vfloat3 operator OP##=(vfloat3 const &that) {                                \
    this->x OP## = that.x;                                                     \
    this->y OP## = that.y;                                                     \
    this->z OP## = that.z;                                                     \
    return *this;                                                              \
  }
  VOP(+)
  VOP(-)
  VOP(*)
  VOP(/)
#undef VOP
#define VOP(OP)                                                                \
  vfloat3 operator OP##=(vfloat const &that) {                                 \
    this->x OP## = that;                                                       \
    this->y OP## = that;                                                       \
    this->z OP## = that;                                                       \
    return *this;                                                              \
  }
  VOP(+)
  VOP(-)
  VOP(*)
  VOP(/)
#undef VOP
#define VOP(OP)                                                                \
  vfloat3 operator OP##=(f32 that) {                                           \
    this->x OP## = that;                                                       \
    this->y OP## = that;                                                       \
    this->z OP## = that;                                                       \
    return *this;                                                              \
  }
  VOP(+)
  VOP(-)
  VOP(*)
  VOP(/)
#undef VOP
  //
  vfloat dot(vfloat3 const &that) const {
    vfloat rx, ry, rz;
    rx        = this->x * that.x;
    ry        = this->y * that.y;
    rz        = this->z * that.z;
    vfloat rs = rx + ry + rz;
    return rs;
  }
  vfloat length2() const { return dot(*this); }
  vfloat ilength() const {
    vfloat r = length2();
    r.val    = _mm512_maskz_rsqrt14_ps(cur_vmask(), r.val);
    return r;
  }
  vfloat length() const {
    vfloat r = length2();
    r.val    = _mm512_maskz_sqrt_ps(cur_vmask(), r.val);
    return r;
  }
  vfloat3 operator-() const {
    vfloat3 r;
    r.x = -this->x;
    r.y = -this->y;
    r.z = -this->z;
    return r;
  }
  vfloat3 normalize() const {
    vfloat  ilen = ilength();
    vfloat3 r;
    r.x = this->x * ilen;
    r.y = this->y * ilen;
    r.z = this->z * ilen;
    return r;
  }
  static vfloat3 splat(f32 x) {
    vfloat3 r;
    r.x = vfloat::splat(x);
    r.y = vfloat::splat(x);
    r.z = vfloat::splat(x);
    return r;
  }
  static vfloat3 splat(f32 x, f32 y, f32 z) {
    vfloat3 r;
    r.x = vfloat::splat(x);
    r.y = vfloat::splat(y);
    r.z = vfloat::splat(z);
    return r;
  }
  static vfloat3 blend(vfloat3 const &a, vfloat3 const &b, vbool const &k) {
    vfloat3 r;
    r.x = vfloat::blend(a.x, b.x, k);
    r.y = vfloat::blend(a.y, b.y, k);
    r.z = vfloat::blend(a.z, b.z, k);
    return r;
  }
  static inline vfloat3 splat(float3 const &v) {
    return vfloat3::splat(v.x, v.y, v.z);
  }
  void dump() const {
    fprintf(stdout, "vfloat3:\n");
    fprintf(stdout, "x:\n");
    ito(16) fprintf(stdout, " %f ", x[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "y:\n");
    ito(16) fprintf(stdout, " %f ", y[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "z:\n");
    ito(16) fprintf(stdout, " %f ", z[i]);
    fprintf(stdout, "\n");
  }
  float3 extract(u32 i) const { return float3{x[i], y[i], z[i]}; }
};

static inline vfloat vdot(vfloat3 const &a, vfloat3 const &b) {
  return a.dot(b);
}

static inline vfloat vsign(vfloat const &a) {
  return vfloat::blend(vfloat::splat(1.0f), vfloat::splat(-1.0f), a < 0.0f);
}

static inline vfloat vmax(vfloat const &a, vfloat const &b) {
  return vfloat::blend(a, b, a < b);
}

static inline vfloat vmin(vfloat const &a, vfloat const &b) {
  return vfloat::blend(a, b, a > b);
}

static inline vfloat vmax3(vfloat const &a, vfloat const &b, vfloat const &c) {
  return vmax(a, vmax(b, c));
}

static inline vfloat vmin3(vfloat const &a, vfloat const &b, vfloat const &c) {
  return vmin(a, vmin(b, c));
}

static inline vfloat3 vcross(vfloat3 const &a, vfloat3 const &b) {
  // a.yzx * b.zxy - a.zxy * b.yzx
  vfloat3 out;
  out.x = a.y * b.z - a.z * b.y;
  out.y = a.z * b.x - a.x * b.z;
  out.z = a.x * b.y - a.y * b.x;
  return out;
}

#define SOP(OP)                                                                \
  static inline vfloat3 operator OP(f32 a, vfloat3 const &vf) {                \
    vfloat      va = vfloat::splat(a);                                         \
    vfloat3     r;                                                             \
    r.x = va OP vf.x;                                                          \
    r.y = va OP vf.y;                                                          \
    r.z = va OP vf.z;                                                          \
    return r;                                                                  \
  }
SOP(/)
SOP(*)
SOP(+)
SOP(-)
#undef SOP
#define SOP(OP)                                                                \
  static inline vfloat3 operator OP(vfloat3 const &vf, f32 a) {                \
    vfloat        va = vfloat::splat(a);                                       \
    vfloat3       r;                                                           \
    r.x = vf.x OP va;                                                          \
    r.y = vf.y OP va;                                                          \
    r.z = vf.z OP va;                                                          \
    return r;                                                                  \
  }
SOP(/)
SOP(*)
SOP(+)
SOP(-)
#undef SOP
#define SOP(OP)                                                                \
  static inline vfloat3 operator OP(vfloat va, vfloat3 const &vf) {            \
    vfloat3     r;                                                             \
    r.x = va OP vf.x;                                                          \
    r.y = va OP vf.y;                                                          \
    r.z = va OP vf.z;                                                          \
    return r;                                                                  \
  }
SOP(/)
SOP(*)
SOP(+)
SOP(-)
#undef SOP
#define SOP(OP)                                                                \
  static inline vfloat3 operator OP(vfloat3 const &vf, vfloat va) {            \
    vfloat3       r;                                                           \
    r.x = vf.x OP va;                                                          \
    r.y = vf.y OP va;                                                          \
    r.z = vf.z OP va;                                                          \
    return r;                                                                  \
  }
SOP(/)
SOP(*)
SOP(+)
SOP(-)
#undef SOP
//#endif // UTILS_AVX512

struct vRay {
  vfloat3 o;
  vfloat3 d;
};

struct Collision {
  u32    mesh_id, face_id;
  float3 position;
  float3 normal;
  float  t, u, v;
};

struct vCollision {
  vint      mesh_id, face_id;
  vfloat3   position;
  vfloat3   normal;
  vfloat    t, u, v;
  Collision extract(u32 i) {
    Collision col;
    col.mesh_id  = mesh_id[i];
    col.face_id  = face_id[i];
    col.position = position.extract(i);
    col.normal   = normal.extract(i);
    col.t        = t[i];
    col.v        = v[i];
    col.u        = u[i];
    return col;
  }
};

// Möller–Trumbore intersection algorithm
static bool ray_triangle_test_moller(vec3 ray_origin, vec3 ray_dir, vec3 v0,
                                     vec3 v1, vec3 v2,
                                     Collision &out_collision) {
  ZoneScopedS(16);
  float invlength = 1.0f / std::sqrt(glm::dot(ray_dir, ray_dir));
  ray_dir *= invlength;

  const float EPSILON = 1.0e-6f;
  vec3        edge1, edge2, h, s, q;
  float       a, f, u, v;
  edge1 = v1 - v0;
  edge2 = v2 - v0;
  h     = glm::cross(ray_dir, edge2);
  a     = glm::dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return false; // This ray is parallel to this triangle.
  f = 1.0 / a;
  s = ray_origin - v0;
  u = f * glm::dot(s, h);
  if (u < 0.0 || u > 1.0) return false;
  q = glm::cross(s, edge1);
  v = f * glm::dot(ray_dir, q);
  if (v < 0.0 || u + v > 1.0) return false;
  // At this stage we can compute t to find out where the intersection point
  // is on the line.
  float t = f * glm::dot(edge2, q);
  if (t > EPSILON) // ray intersection
  {
    out_collision.t      = t * invlength;
    out_collision.u      = u;
    out_collision.v      = v;
    out_collision.normal = glm::normalize(cross(edge1, edge2));
    out_collision.normal *= sign(-glm::dot(ray_dir, out_collision.normal));
    out_collision.position = ray_origin + ray_dir * t;

    return true;
  } else // This means that there is a line intersection but not a ray
         // intersection.
    return false;
}

static inline vbool vray_triangle_test_moller(vfloat3 ray_origin,
                                              vfloat3 ray_dir, vfloat3 v0,
                                              vfloat3 v1, vfloat3 v2,
                                              vCollision &out_collision) {
  ZoneScopedS(16);
  // vfloat invlength = ray_dir.ilength();
  // ray_dir          = ray_dir * invlength;

  const float EPSILON = 1.0e-6f;
  vfloat3     edge1, edge2, h, s, q;
  vfloat      a, f, u, v;
  edge1 = v1 - v0;
  edge2 = v2 - v0;
  h     = vcross(ray_dir, edge2);
  a     = vdot(edge1, h);

  vbool vmsk = cur_vmask();

  vmsk = vmsk && !(a > -EPSILON && a < EPSILON);
  if (vmsk.none()) return vmsk;

  f = 1.0 / a;
  s = ray_origin - v0;
  u = f * vdot(s, h);

  vmsk = vmsk && !(u < 0.0f || u > 1.0f);
  if (vmsk.none()) return vmsk;

  q = vcross(s, edge1);
  v = f * vdot(ray_dir, q);

  vmsk = vmsk && !(v < 0.0 || u + v > 1.0);
  if (vmsk.none()) return vmsk;

  // At this stage we can compute t to find out where the intersection point
  // is on the line.
  vfloat t = f * vdot(edge2, q);

  vmsk = vmsk && t > EPSILON;
  if (vmsk.none()) return vmsk;

  out_collision.t      = t; // * invlength;
  out_collision.u      = u;
  out_collision.v      = v;
  out_collision.normal = vcross(edge1, edge2).normalize();
  out_collision.normal *= vsign(-vdot(ray_dir, out_collision.normal));
  out_collision.position = ray_origin + ray_dir * t;

  return vmsk;
}

// Woop intersection algorithm
static bool ray_triangle_test_woop(vec3 ray_origin, vec3 ray_dir, vec3 a,
                                   vec3 b, vec3 c, Collision &out_collision) {
  const float EPSILON        = 1.0e-4f;
  vec3        ab             = b - a;
  vec3        ac             = c - a;
  vec3        n              = cross(ab, ac);
  mat4        world_to_local = glm::inverse(mat4(
      //
      ab.x, ab.y, ab.z, 0.0f,
      //
      ac.x, ac.y, ac.z, 0.0f,
      //
      n.x, n.y, n.z, 0.0f,
      //
      a.x, a.y, a.z, 1.0f
      //
      ));
  vec4        ray_origin_local =
      (world_to_local * vec4(ray_origin.x, ray_origin.y, ray_origin.z, 1.0f));
  vec4 ray_dir_local =
      world_to_local * vec4(ray_dir.x, ray_dir.y, ray_dir.z, 0.0f);
  if (std::abs(ray_dir_local.z) < EPSILON) return false;
  float t = -ray_origin_local.z / ray_dir_local.z;
  if (t < EPSILON) return false;
  float u = ray_origin_local.x + t * ray_dir_local.x;
  float v = ray_origin_local.y + t * ray_dir_local.y;
  if (u > 0.0f && v > 0.0f && u + v < 1.0f) {
    out_collision.t        = t;
    out_collision.u        = u;
    out_collision.v        = v;
    out_collision.normal   = glm::normalize(n) * sign(-ray_dir_local.z);
    out_collision.position = ray_origin + ray_dir * t;
    return true;
  }
  return false;
}

struct Ray {
  float3 o;
  float3 d;
};

struct Tri {
  u32    id;
  float3 a;
  float3 b;
  float3 c;
  void   get_aabb(float3 &min, float3 &max) const {
    ito(3) min[i] = MIN(a[i], MIN(b[i], c[i]));
    ito(3) max[i] = MAX(a[i], MAX(b[i], c[i]));
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
};

static_assert(sizeof(Tri) == 40, "Blamey!");

struct vTri {
  vint    id;
  vfloat3 a;
  vfloat3 b;
  vfloat3 c;
  void    store(u32 i, Tri const &tri) {
    a.x[i] = tri.a.x;
    a.y[i] = tri.a.y;
    a.z[i] = tri.a.z;
    b.x[i] = tri.b.x;
    b.y[i] = tri.b.y;
    b.z[i] = tri.b.z;
    c.x[i] = tri.c.x;
    c.y[i] = tri.c.y;
    c.z[i] = tri.c.z;
    id[i]  = tri.id;
  }
  Tri extract(u32 i) const {
    Tri out;
    out.id = id[i];
    out.a  = a.extract(i);
    out.b  = b.extract(i);
    out.c  = c.extract(i);
    return out;
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
  vbool vinside(vfloat3 tmin) {
    return                 //
        tmin.x >= min.x && //
        tmin.x <= max.x && //
        tmin.y >= min.y && //
        tmin.y <= max.y && //
        tmin.z >= min.z && //
        tmin.z <= max.z;
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
  vbool vintersects_ray(vfloat3 vro, vfloat3 vrd) {
#if 0
    vbool vmsk = cur_vmask();
    ito(16) {
      if (vmsk.is_enabled(i)) {
        vmsk.set(i, intersects_ray(vro.extract(i), vrd.extract(i)));
      }
    }
    return vmsk;
#else
    vfloat3 vinvd = vfloat3::splat(1.0f, 1.0f, 1.0f) / vrd;
    vfloat  vdx_n = (min.x - vro.x) * vinvd.x;
    vfloat  vdy_n = (min.y - vro.y) * vinvd.y;
    vfloat  vdz_n = (min.z - vro.z) * vinvd.z;
    vfloat  vdx_f = (max.x - vro.x) * vinvd.x;
    vfloat  vdy_f = (max.y - vro.y) * vinvd.y;
    vfloat  vdz_f = (max.z - vro.z) * vinvd.z;
    vfloat  vnt =
        vmax3(vmin(vdx_n, vdx_f), vmin(vdy_n, vdy_f), vmin(vdz_n, vdz_f));
    vfloat vft =
        vmin3(vmax(vdx_n, vdx_f), vmax(vdy_n, vdy_f), vmax(vdz_n, vdz_f));
    vft -= EPS;
    vbool ret = vnt < vft || vinside(vro);
    return ret;
#endif
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

struct BVH_Helper {
  float3      min;
  float3      max;
  Array<Tri>  tris;
  BVH_Helper *left;
  BVH_Helper *right;
  bool        is_leaf;
  void        init() {
    MEMZERO(*this);
    tris.init();
    min     = float3(1.0e10f, 1.0e10f, 1.0e10f);
    max     = float3(-1.0e10f, -1.0e10f, -1.0e10f);
    is_leaf = true;
  }
  void release() {
    if (left != NULL) left->release();
    if (right != NULL) right->release();
    tris.release();
    MEMZERO(*this);
    delete this;
  }
  void reserve(size_t size) { tris.reserve(size); }
  void push(Tri const &tri) {
    ZoneScoped;

    tris.push(tri);
    float3 tmin, tmax;
    tri.get_aabb(tmin, tmax);
    ito(3) min[i] = MIN(min[i], tmin[i]);
    ito(3) max[i] = MAX(max[i], tmax[i]);
  }
  u32 split(u32 max_items, u32 depth = 0) {
    ZoneScoped;

    ASSERT_DEBUG(depth < BVH_Node::MAX_DEPTH);
    if (tris.size > max_items && depth < BVH_Node::MAX_DEPTH) {
      left = new BVH_Helper;
      left->init();
      left->reserve(tris.size / 2);
      right = new BVH_Helper;
      right->init();
      right->reserve(tris.size / 2);
      struct Sorting_Node {
        u32   id;
        float val;
      };
      {
        TMP_STORAGE_SCOPE;
        u32           num_items = tris.size;
        Sorting_Node *sorted_dims[6];
        ito(6) sorted_dims[i] =
            (Sorting_Node *)tl_alloc_tmp(sizeof(Sorting_Node) * num_items);
        Tri *items = tris.ptr;
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
          u32 tri_id = sorted_dims[max_dim_id][i].id;
          Tri tri    = tris[tri_id];
          if (i < split_index) {
            left->push(tri);
          } else {
            right->push(tri);
          }
        }
      }
      is_leaf = false;
      tris.release();
      u32 cnt = left->split(max_items, depth + 1);
      cnt += right->split(max_items, depth + 1);
      return cnt + 1;
    }
    return 1;
  }
};

static_assert(sizeof(BVH_Node) == 28, "Blamey!");

struct BVH {
  Array<vTri>     tri_pool;
  Array<BVH_Node> node_pool;
  BVH_Node *      root;

  void gen(BVH_Node *node, BVH_Helper *hnode) {
    ZoneScoped;

    ASSERT_ALWAYS(node != NULL);
    ASSERT_ALWAYS(hnode != NULL);
    if (hnode->is_leaf) {
      ASSERT_DEBUG(hnode->tris.size != 0);
      u32 tri_offset = alloc_tri_chunk();
      node->init_leaf(hnode->min, hnode->max, tri_offset);
      ASSERT_DEBUG(hnode->tris.size <= BVH_Node::MAX_ITEMS);
      node->set_num_items(hnode->tris.size);
      vTri *tris = tri_pool.at(node->items_offset());
      ito(hnode->tris.size) { tris->store(i, hnode->tris[i]); }
    } else {
      BVH_Node *children = node_pool.alloc(2);
      node->init_branch(hnode->min, hnode->max, children);
      gen(children + 0, hnode->left);
      gen(children + 1, hnode->right);
    }
  }
  void init(Tri *tris, u32 num_tris) { //
    ZoneScopedN("BVH::init()");

    BVH_Helper *hroot = new BVH_Helper;
    hroot->init();
    hroot->reserve(num_tris);
    defer(hroot->release());
    ito(num_tris) { hroot->push(tris[i]); }
    u32 ncnt = hroot->split(BVH_Node::MAX_ITEMS);
    tri_pool.init();
    node_pool.init();
    tri_pool.reserve(num_tris * 4);
    node_pool.reserve(ncnt);
    root = node_pool.alloc(1);
    gen(root, hroot);
  }
  u32 alloc_tri_chunk() {
    vTri *new_chunk = tri_pool.alloc(1);
    MEMZERO(*new_chunk);
    vTri *tri_root = tri_pool.at(0);
    return (u32)(((u8 *)new_chunk - (u8 *)tri_root) / sizeof(vTri));
  }
  void release() {
    tri_pool.release();
    node_pool.release();
  }
  template <typename F> void traverse(float3 ro, float3 rd, F fn) {
    if (!root->intersects_ray(ro, rd)) return;
    traverse(root, ro, rd, fn);
  }
  template <typename F>
  void traverse(BVH_Node *node, float3 ro, float3 rd, F fn) {
    ZoneScoped;
    if (node->is_leaf()) {
      vTri *tris     = tri_pool.at(node->items_offset());
      u32   num_tris = node->num_items();
      ASSERT_ALWAYS(num_tris <= vfloat3::WIDTH);
      fn(*tris);
    } else {
      BVH_Node *children = node->first_child();
      BVH_Node *left     = children + 0;
      BVH_Node *right    = children + 1;
      if (left->intersects_ray(ro, rd)) traverse(left, ro, rd, fn);
      if (right->intersects_ray(ro, rd)) traverse(right, ro, rd, fn);
    }
  }
  template <typename F> void vtraverse(vfloat3 ro, vfloat3 rd, F fn) {
    mask().set(root->vintersects_ray(ro, rd));
    if (mask().cur().none()) return;
    vtraverse(root, ro, rd, fn);
  }
  template <typename F>
  void vtraverse(BVH_Node *node, vfloat3 ro, vfloat3 rd, F fn) {
    ZoneScoped;
    if (node->is_leaf()) {
      vTri *tris     = tri_pool.at(node->items_offset());
      u32   num_tris = node->num_items();
      ASSERT_ALWAYS(num_tris <= vfloat3::WIDTH);
      fn(*tris, num_tris);
    } else {
      BVH_Node *children = node->first_child();
      BVH_Node *left     = children + 0;
      BVH_Node *right    = children + 1;


      vbool     vmask    = mask().cur();

      mask().set(left->vintersects_ray(ro, rd));
      if (mask().cur().any()) vtraverse(left, ro, rd, fn);
      mask().set(vmask);

      mask().set(right->vintersects_ray(ro, rd));
      if (mask().cur().any()) vtraverse(right, ro, rd, fn);
      mask().set(vmask);
    }
  }
};

void vec_test() {
  mask().enable_all();
  // mask().disable(0);
  // mask().disable(5);
  //{
  //  vfloat a = vfloat::splat(1.0f);
  //  vfloat b = vfloat::splat(2.0f);
  //  vfloat c = a + b;
  //  mask().dump();
  //  c.dump();
  //}
  //{
  //  vfloat3 a = vfloat3::splat(1.0f, 2.0f, 3.0f);
  //  vfloat3 b = vfloat3::splat(-2.0f, -3.0f, 5.0f);

  //  vfloat3 c = (1.0f + a + b * 2.0f).normalize();
  //  c.dump();
  //  c.x = vsign(c.x);
  //  c.y = vsign(c.y);
  //  c.z = vsign(c.z);
  //  c.dump();
  //}
  //{
  //  vfloat3 a = vfloat3::splat(1.0f, 0.0f, 0.0f);
  //  vfloat3 b = vfloat3::splat(0.0f, 4.0f, 0.0f);
  //  vfloat3 c = vcross(a, b).normalize();
  //  c.dump();
  //}
  {
    vfloat a;
    kto(16) { a[k] = (f32)k; }
    vfloat b = vfloat::splat(5.0f);
    a.dump();
    b.dump();
    (a + b).dump();
    (a < b).dump();
    (a > b).dump();
    (a == b).dump();
    (a > b || a == b || a < b).dump();
  }
}

void sort_test() {
  int arr[] = {
      27, 12,  3,  32, 46, 97, 32, 60, 56, 91, 69, 76, 7,  95, 25, 86, 96,
      5,  88,  88, 42, 30, 35, 74, 93, 28, 82, 23, 98, 31, 22, 55, 53, 23,
      45, 78,  46, 97, 34, 63, 60, 91, 99, 58, 73, 53, 75, 63, 88, 32, 66,
      13, 100, 44, 22, 37, 23, 29, 70, 51, 51, 76, 66, 15, 39, 48, 85, 13,
      89, 97,  17, 36, 41, 75, 92, 43, 29, 82, 31, 59, 67, 26, 49, 38, 20,
      2,  98,  70, 31, 9,  79, 48, 11, 4,  44, 1,  49, 73, 90, 70,
  };
  kto(1000) {
    quicky_sort(arr, ARRAYSIZE(arr), [](int a, int b) { return a < b; });
    ito(ARRAYSIZE(arr) - 1) { ASSERT_ALWAYS(arr[i] <= arr[i + 1]); }
    jto(ARRAYSIZE(arr)) arr[j] = rand();
  }
}

struct Camera {
  float3 position;
  float3 look;
  float3 up;
  float3 right;
  float  fov;
};
Camera gen_camera(float phi, float theta, float r, float3 lookat, float fov) {
  Camera cam;
  cam.position =
      r * float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
  cam.look = -normalize(cam.position);
  cam.position += lookat;
  cam.right = -normalize(cross(cam.look, float3(0.0, 1.0, 0.0)));
  cam.up    = -cross(cam.right, cam.look);
  cam.fov   = fov;
  return cam;
}
Ray gen_ray(Camera cam, float2 uv) {
  Ray r;
  r.o = cam.position;
  r.d = normalize(cam.look + cam.fov * (cam.right * uv.x + cam.up * uv.y));
  return r;
}

inline void nop() {
#if defined(_WIN32)
  __noop();
#else
  __asm__ __volatile__("nop");
#endif
}
inline static uint64_t get_thread_id() {
  auto id = std::this_thread::get_id();
  return std::hash<std::thread::id>()(id);
}
struct Spin_Lock {
  std::atomic<u32> rw_flag;
  void             lock() {
    ZoneScopedS(16);
    u32 expected = 0;
    while (!rw_flag.compare_exchange_strong(expected, 1)) {
      expected = 0;
      while (rw_flag.load() != 0) {
        ito(16) nop(); // yield
      }
    }
  }
  void unlock() { rw_flag.store(0); }
};

// Poor man's queue
// Not thread safe in all scenarios but kind of works in mine
// @Cleanup
template <typename Job_t> struct Queue {
  template <typename Job_t> struct Batch {
    Job_t *          job_queue;
    u32              capacity;
    std::atomic<u32> head;
    Spin_Lock        spinlock;
    void             lock() { spinlock.lock(); }
    void             unlock() { spinlock.unlock(); }
    void             init() {
      ZoneScoped;
      head      = 0;
      capacity  = 1 << 18;
      job_queue = (Job_t *)tl_alloc(sizeof(Job_t) * capacity);
    }
    void release() { tl_free(job_queue); }

    bool has_items() { return head.load() != 0; }
    bool try_dequeue(Job_t &job) {
      ZoneScoped;
      lock();
      defer(unlock());
      if (!has_items()) return false;
      u32 old_head = head.fetch_sub(1);
      job          = job_queue[old_head - 1];
      return true;
    }
    bool try_dequeue(Job_t *jobs, u32 *max_size) {
      ZoneScoped;
      lock();
      defer(unlock());
      if (!has_items()) {
        *max_size = 0;
        return false;
      }
      u32 num_items = MIN(*max_size, head.load());
      u32 old_head  = head.fetch_sub(num_items);
      memcpy(jobs, job_queue + old_head - num_items, sizeof(Job_t) * num_items);
      *max_size = num_items;
      return true;
    }
    void dequeue(Job_t *out, u32 &count) {
      ZoneScoped;
      lock();
      defer(unlock());
      if (head < count) {
        count = head;
      }
      u32 old_head = head.fetch_sub(count);
      memcpy(out, job_queue + head, count * sizeof(out[0]));
    }
    void enqueue(Job_t job) {
      ZoneScoped;
      lock();
      defer(unlock());
      ASSERT_PANIC(!std::isnan(job.ray_dir.x) && !std::isnan(job.ray_dir.y) &&
                   !std::isnan(job.ray_dir.z));
      u32 old_head        = head.fetch_add(1);
      job_queue[old_head] = job;
      ASSERT_PANIC(head <= capacity);
    }
    void enqueue(Job_t const *jobs, u32 num) {
      ZoneScoped;
      u32 old_head = 0;
      {
        lock();
        defer(unlock());
        old_head = head.fetch_add(num);
        ASSERT_PANIC(head <= capacity);
      }
      memcpy(job_queue + old_head, jobs, num * sizeof(Job_t));
    }
    bool has_job() { return head != 0u; }
    void reset() { head = 0u; }
  };

  static constexpr u32 NUM_BATCHES = 512;
  Batch<Job_t>         batches[NUM_BATCHES];
  void                 init() {
    ZoneScoped;
    ito(NUM_BATCHES) { batches[i].init(); }
  }
  void release() {
    ZoneScoped;
    ito(NUM_BATCHES) { batches[i].release(); }
  }
  bool has_items_any() {
    ito(NUM_BATCHES) if (batches[i].has_items()) return true;
    return false;
  }
  bool has_items(u32 id) { return batches[id % NUM_BATCHES].has_items(); }
  bool try_dequeue(u32 id, Job_t &job) {
    return batches[id % NUM_BATCHES].try_dequeue(job);
  }
  bool try_dequeue(u32 id, Job_t *jobs, u32 *max_size) {
    return batches[id % NUM_BATCHES].try_dequeue(jobs, max_size);
  }
  void dequeue(u32 id, Job_t *out, u32 &count) {
    return batches[id % NUM_BATCHES].dequeue(out, count);
  }
  void enqueue(u32 id, Job_t job) {
    return batches[id % NUM_BATCHES].enqueue(job);
  }
  void enqueue(u32 id, Job_t const *jobs, u32 num) {
    return batches[id % NUM_BATCHES].enqueue(jobs, num);
  }
  bool has_job(u32 id) { return batches[id % NUM_BATCHES].has_job(); }
};

struct Scene {
  // 3D model + materials
  PBR_Model   model;
  Array<BVH>  bvhs;
  Image2D_Raw env_spheremap;

  void init(string_ref filename, string_ref env_filename) {
    ZoneScopedN("Scene::init()");

    model         = load_gltf_pbr(filename);
    env_spheremap = load_image(env_filename);
    bvhs.init();
    Array<Tri> tri_pool;
    tri_pool.init();
    defer(tri_pool.release());
    kto(model.meshes.size) {
      Raw_Mesh_Opaque &mesh = model.meshes[k];
      tri_pool.reset();
      Tri *tris     = tri_pool.alloc(mesh.num_indices / 3);
      u32  num_tris = mesh.num_indices / 3;
      ito(num_tris) {
        Triangle_Full ftri = mesh.fetch_triangle(i);
        tris[i].a          = ftri.v0.position;
        tris[i].b          = ftri.v1.position;
        tris[i].c          = ftri.v2.position;
        tris[i].id         = i;
      }
      BVH bvh;
      bvh.init(tris, num_tris);
      bvhs.push(bvh);
    }
  }

  float4 env_value(float3 ray_dir, float3 color) {
    if (env_spheremap.data == NULL) {
      return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float  theta = std::acos(ray_dir.y);
    float2 xy    = normalize(float2(ray_dir.z, -ray_dir.x));
    float  phi   = -std::atan2(xy.x, xy.y);
    return float4(color, 1.0f) *
           env_spheremap.sample(float2((phi / PI / 2.0f) + 0.5f, theta / PI));
  }

  bool collide(float3 ro, float3 rd, Collision &col) {
    ZoneScoped;

    col.t    = FLT_MAX;
    bool hit = false;
    kto(model.meshes.size) {
      ZoneScoped;

      BVH &bvh = bvhs[k];
      bvh.traverse(ro, rd, [&](vTri const &tri) {
        ZoneScoped;
#if 1
        vCollision vc;
        vc.t          = vfloat::splat(FLT_MAX);
        vbool cur_hit = vray_triangle_test_moller(
            vfloat3::splat(ro), vfloat3::splat(rd), tri.a, tri.b, tri.c, vc);
        if (cur_hit.none()) return col.t;
        ito(16) {
          if (!cur_hit.is_enabled(i)) continue;
          Tri ntri = tri.extract(i);
          if (vc.t[i] < col.t) {
            hit         = true;
            col         = vc.extract(i);
            col.mesh_id = k;
            col.face_id = ntri.id;
          }
        }
        return col.t;
#else
        Collision c;
        c.t = FLT_MAX;
        ito(16) {
          Tri ntri = tri.extract(i);
          if (ray_triangle_test_moller(ro, rd, ntri.a, ntri.b, ntri.c, c)) {
            if (c.t < col.t) {
              hit         = true;
              col         = c;
              col.mesh_id = k;
              col.face_id = ntri.id;
            }
          }
        }
        return c.t;
#endif
      });
    }
    return hit;
  }

  vbool vcollide(vfloat3 vro, vfloat3 vrd, vCollision &col) {
    ZoneScoped;
    col.t = vfloat::splat(FLT_MAX);
    vbool hit{0};
    vbool vmsk = cur_vmask();

    kto(model.meshes.size) {
      ZoneScoped;
      BVH &bvh = bvhs[k];
      mask().set(vmsk);
      bvh.vtraverse(vro, vrd, [&](vTri const &tri, u32 num_triangles) {
        ZoneScoped;
        vCollision c;
        c.t        = vfloat::splat(FLT_MAX);
        vbool vmsk = cur_vmask();
        if (vmsk.popcnt() == 1 && num_triangles == 1) {
          ZoneScopedN("1:1 test");
          u32 ray_id;
          vmsk.lsb(ray_id);
          float3    ro = vro.extract(ray_id);
          float3    rd = vrd.extract(ray_id);
          Collision c;
          c.t      = FLT_MAX;
          Tri ntri = tri.extract(0);
          if (ray_triangle_test_moller(ro, rd, ntri.a, ntri.b, ntri.c, c)) {
            if (c.t < col.t) {
              hit.enable(ray_id);
              /*col         = c;
              col.mesh_id = k;
              col.face_id = ntri.id;*/
            }
          }
        } else {
          ZoneScopedN("1:16 test");
          u32   i    = 0;
          vbool iter = vmsk;
          while (iter.take_lsb(i)) {
            float3 ro = vro.extract(i);
            float3 rd = vrd.extract(i);
            mask().enable_all();
            vbool cur_hit = vray_triangle_test_moller(
                vfloat3::splat(ro), vfloat3::splat(rd), tri.a, tri.b, tri.c, c);
            mask().set(vmsk);
            if (cur_hit.none()) continue;
            hit.enable(i);
            /*jto(16) {
              if (!cur_hit.is_enabled(i)) continue;
              if (c.t[j] < col.t[i]) {
                hit.enable(i);
                col.mesh_id[i] = c.mesh_id[j];
                col.face_id[i] = c.face_id[j];
              }
            }*/
          }
        }
      });
      // hit = hit | cur_vmask();
    }
    return hit;
  }

  void release() {
    ito(bvhs.size) bvhs[i].release();
    bvhs.release();
    env_spheremap.release();
    model.release();
  }
};

uint32_t rgba32f_to_rgba8_unorm(float r, float g, float b, float a) {
  uint8_t r8 = (uint8_t)(clamp(r, 0.0f, 1.0f) * 255.0f);
  uint8_t g8 = (uint8_t)(clamp(g, 0.0f, 1.0f) * 255.0f);
  uint8_t b8 = (uint8_t)(clamp(b, 0.0f, 1.0f) * 255.0f);
  uint8_t a8 = (uint8_t)(clamp(a, 0.0f, 1.0f) * 255.0f);
  return                     //
      ((uint32_t)r8 << 0) |  //
      ((uint32_t)g8 << 8) |  //
      ((uint32_t)b8 << 16) | //
      ((uint32_t)a8 << 24);  //
}

uint32_t rgba32f_to_srgba8_unorm(float r, float g, float b, float a) {
  uint8_t r8 = (uint8_t)(clamp(std::pow(r, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  uint8_t g8 = (uint8_t)(clamp(std::pow(g, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  uint8_t b8 = (uint8_t)(clamp(std::pow(b, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  uint8_t a8 = (uint8_t)(clamp(std::pow(a, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f);
  return                     //
      ((uint32_t)r8 << 0) |  //
      ((uint32_t)g8 << 8) |  //
      ((uint32_t)b8 << 16) | //
      ((uint32_t)a8 << 24);  //
}

struct RTScene {
  struct Path_Tracing_Job {
    float3 ray_origin;
    float3 ray_dir;
    // Color weight applied to the sampled light
    vec3 color;
    u32  pixel_x, pixel_y;
    f32  weight;
    // For visibility checks
    u32 light_id;
    u32 depth,
        // Used to track down bugs
        _depth;
  };

  struct Per_HW_Thread {
    Random_Factory          rfs;
    Array<Path_Tracing_Job> local_queue;
    void                    init() { local_queue.init(); }
    void                    release() { local_queue.release(); }
  };
  Per_HW_Thread           phw[64];
  Scene                   scene;
  u32                     jobs_per_item     = 8 * 32;
  bool                    use_jobs          = true;
  u32                     max_jobs_per_iter = 1 << 20;
  Queue<Path_Tracing_Job> queue;
  Array<float4>           rt0;
  int2                    iResolution = int2(512, 512);
  Camera                  cam;
  float2                  halton_cache[0x100];
  u32                     primary_rays_cnt = 0;
  Spin_Lock               rt0_locks[0x400];
  u32                     num_cpus;

  void retire_rt0(u32 i, u32 j, float4 d) {
    Spin_Lock &lock =
        rt0_locks[(hash_of((u64)i) ^ hash_of((u64)j)) % ARRAY_SIZE(rt0_locks)];
    lock.lock();
    defer(lock.unlock());
    rt0[i * iResolution.x + j] += d;
  }
  void trace_primary(u32 batch, u32 i, u32 j, float3 ro, float3 rd) {
    Path_Tracing_Job job;
    job.color      = float3(1.0f, 1.0f, 1.0f);
    job.depth      = 0;
    job._depth     = 0;
    job.light_id   = 0;
    job.pixel_x    = j;
    job.pixel_y    = i;
    job.ray_dir    = rd;
    job.ray_origin = ro;
    job.weight     = 1.0f;
    queue.enqueue(batch, job);
  }

  void push_primary_rays(u32 num) {
    ZoneScopedN("Pushing primary rays");
    // js.queue.reserve(iResolution.y * iResolution.x * 128);
    ito(iResolution.y) {
      jto(iResolution.x) {
        kto(num) {
          float2 jitter =
              halton_cache[(primary_rays_cnt + k) % ARRAY_SIZE(halton_cache)];
          float2 uv = float2((float(j) + jitter.x) / iResolution.y,
                             (float(iResolution.y - i - 1) + jitter.y) /
                                 iResolution.y) *
                          2.0f -
                      1.0f;
          Ray ray        = gen_ray(cam, uv);
          u8  intersects = 0;
          trace_primary((u32)(hash_of(i) ^ hash_of(j) ^ hash_of(k / 16)), i, j,
                        ray.o, ray.d);
        }
      }
    }
    primary_rays_cnt += num;
  }

  void init() {
    scene.init(stref_s("models/human_bust_sculpt/scene.gltf"),
               // scene.init(stref_s("models/old_tree/scene.gltf"),
               stref_s("env/kloetzle_blei_2k.hdr"));
    queue.init();
    rt0.init();
    rt0.resize(iResolution.x * iResolution.y);
    rt0.memzero();

    num_cpus = std::thread::hardware_concurrency();
#if defined(_WIN32)
    {
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      num_cpus = sysinfo.dwNumberOfProcessors;
    }
#endif

    float2      m  = float2(0.0f, 0.0f);
    const float PI = 3.141592654f;
    cam = gen_camera(PI * 0.3f, PI * 0.5, 45.0, float3(0.0, 1.0, 0.0), 1.0);

    ito(ARRAY_SIZE(phw)) phw[i].init();

    ito(ARRAY_SIZE(halton_cache)) {
      f32 jitter_u    = halton(i + 1, 2);
      f32 jitter_v    = halton(i + 1, 3);
      halton_cache[i] = float2(jitter_u, jitter_v);
    }
  }
  void trace() {
    ZoneScopedN("Trace");
    std::atomic<u32> rays_traced;
    std::atomic<u32> rays_hit;
    std::atomic<u32> rays_misses;
    std::atomic<u32> workers_in_progress;

    bool                    working = true;
    std::mutex              cv_mux;
    std::condition_variable cv;
    std::mutex              finished_mux;
    std::condition_variable finished_cv;
    std::thread *           thpool[0x100];
    fprintf(stdout, "Launching %i threads\n", num_cpus);
    auto t0 = clock();
    for (u32 thread_id = 0; thread_id < num_cpus; thread_id++) {
      thpool[thread_id] = new std::thread([&, thread_id] {
      // Set affinity
#if defined(_WIN32)
        SetThreadAffinityMask(GetCurrentThread(), (u64)1 << (u64)thread_id);
#endif // _WIN32
        u32 batch_hash = thread_id;
        while (working) {
          batch_hash = (u32)hash_of(batch_hash);
          if (queue.has_items(batch_hash) == false) {
            finished_cv.notify_one();
            ito(32) nop();
            if (!working) break;
            continue;
          } // if (queue.has_items(batch_hash) == false)

          workers_in_progress++;
          defer({ workers_in_progress--; });
          ZoneScoped;
          const u32      MAX_SECONDARY = 4;
          Per_HW_Thread *res           = &phw[thread_id % ARRAY_SIZE(phw)];

          constexpr u32    RAY_BATCH_SIZE = 16;
          Path_Tracing_Job jobs[RAY_BATCH_SIZE];
          u32              num_rays = RAY_BATCH_SIZE;
          res->local_queue.reset();
          res->local_queue.reserve(RAY_BATCH_SIZE * MAX_SECONDARY);
          u32 rays_emited = 0;
          if (!queue.try_dequeue(batch_hash, jobs, &num_rays)) continue;
          mask().enable_all();
#if 1
          for (u32 i = num_rays; i < RAY_BATCH_SIZE; i++) {
            mask().disable(i);
          }
          vCollision vcol;
          vfloat3    ray_origins;
          vfloat3    ray_dirs;
          kto(num_rays) {
            ray_origins.x[k] = jobs[k].ray_origin.x;
            ray_origins.y[k] = jobs[k].ray_origin.y;
            ray_origins.z[k] = jobs[k].ray_origin.z;
            ray_dirs.x[k]    = jobs[k].ray_dir.x;
            ray_dirs.y[k]    = jobs[k].ray_dir.y;
            ray_dirs.z[k]    = jobs[k].ray_dir.z;
          }
          vbool collide = scene.vcollide(ray_origins, ray_dirs, vcol);
          kto(num_rays) {
            Path_Tracing_Job job = jobs[k];
            if (collide.is_enabled(k)) {
              retire_rt0(job.pixel_y, job.pixel_x,
                         float4(1.0f, 0.0f, 0.0f, 1.0f));
            } else {
              retire_rt0(job.pixel_y, job.pixel_x,
                         float4(0.0f, 0.0f, 0.0f, 1.0f));
            }
          }
#else
          kto(num_rays) {
            Path_Tracing_Job job = jobs[k];
            rays_traced++;
            Collision col;
            bool collide = scene.collide(job.ray_origin, job.ray_dir, col);
            if (collide) {
              retire_rt0(job.pixel_y, job.pixel_x,
                         float4(1.0f, 0.0f, 0.0f, 1.0f));
              // if (job.depth == 2) {
              //  retire_rt0(job.pixel_y, job.pixel_x,
              //             float4(0.0f, 0.0f, 0.0f, job.weight));
              //  goto next_job;
              //} // if (job.depth == 2)
              // PBR_Model &   model = scene.model;
              // PBR_Material &mat   = model.materials[col.mesh_id];
              // const u32     N     = MAX_SECONDARY >> job.depth;
              // ito(N) {
              //  float3           rn =
              //  res->rfs.sample_lambert_BRDF(col.normal); Path_Tracing_Job
              //  new_job; MEMZERO(new_job); new_job.color      =
              //  float3(1.0f, 1.0f, 1.0f); new_job.depth      = job.depth + 1;
              //  new_job._depth     = 0;
              //  new_job.light_id   = 0;
              //  new_job.pixel_x    = job.pixel_x;
              //  new_job.pixel_y    = job.pixel_y;
              //  new_job.ray_dir    = rn;
              //  new_job.ray_origin = col.position + 1.0e-4f * col.normal;
              //  new_job.weight     = job.weight / N;
              //  res->local_queue.push(new_job);
              //  rays_emited++;
              //} // ito(N)
              rays_hit++;
            } else { // if (collide)
              retire_rt0(job.pixel_y, job.pixel_x,
                         float4(0.0f, 0.0f, 0.0f, 1.0f));
              rays_misses++;
              /* if (job.depth == 0) {
                 retire_rt0(job.pixel_y, job.pixel_x,
                            float4(0.0f, 0.0f, 0.0f, job.weight));
               } else {
                 float4 env = scene.env_value(job.ray_dir, job.color);
                 env.w      = 1.0f;
                 retire_rt0(job.pixel_y, job.pixel_x, job.weight * env);
               }*/
            } // else (collide)

            if (res->local_queue.size != 0)
              queue.enqueue(batch_hash, &res->local_queue[0],
                            res->local_queue.size);
          next_job:
            (void)0;
          } // kto(num_rays)
#endif
        } // while (working)
      });
      cv.notify_one();
    }
    {
      std::unique_lock<std::mutex> lk(finished_mux);
      finished_cv.wait(lk, [&] {
        return queue.has_items_any() == false && workers_in_progress == 0;
      });
      working = false;

      ito(num_cpus) {
        cv.notify_all();
        thpool[i]->join();
        delete thpool[i];
      }
    }
    auto t1 = clock();
    fprintf(stdout,
            "Time        : %f\n"
            "Traced Rays : %i\n"
            "Hit    Rays : %i\n"
            "Missed Rays : %i\n",
            (t1 - t0) * 1.0e-3f, rays_traced.load(), rays_hit.load(),
            rays_misses.load());
  }

  void write_ppm() {
    TMP_STORAGE_SCOPE;
    u8 * rgb_image    = (u8 *)tl_alloc_tmp(iResolution.x * iResolution.y * 3);
    auto retire_final = [&](u32 i, u32 j, float4 d) {
      d /= (d.w + 1.0e-6f);
      // d *= 1.5f;
      u32 rgba8 = rgba32f_to_rgba8_unorm(d.r, d.g, d.b, 1.0f);
      rgb_image[i * iResolution.x * 3 + j * 3 + 0] = (rgba8 >> 0) & 0xffu;
      rgb_image[i * iResolution.x * 3 + j * 3 + 1] = (rgba8 >> 8) & 0xffu;
      rgb_image[i * iResolution.x * 3 + j * 3 + 2] = (rgba8 >> 16) & 0xffu;
    };
    ito(iResolution.y) {
      jto(iResolution.x) { retire_final(i, j, rt0[i * iResolution.x + j]); }
    }
    write_image_2d_i24_ppm("image.ppm", rgb_image, iResolution.x * 3,
                           iResolution.x, iResolution.y);
  }

  void release() {
    rt0.release();
    scene.release();
    ito(ARRAY_SIZE(phw)) phw[i].release();
  }
};

int main(int argc, char *argv[]) {
  ZoneScoped;

  (void)argc;
  (void)argv;
  // vec_test();
  // sort_test();

  RTScene rts;
  rts.init();
  rts.push_primary_rays(64);
  rts.trace();
  rts.write_ppm();
  defer({ rts.release(); });

  //#ifdef UTILS_TL_IMPL_DEBUG
  //  assert_tl_alloc_zero();
  //#endif
  return 0;
}