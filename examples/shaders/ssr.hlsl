#include "include/utils.hlsli"

struct FrameConstants {
  float4x4 world_to_view;
  float4x4 view_to_world;
  float4x4 view_to_proj;
  float4x4 proj_to_view;
  float3   camera_pos;
  float3   camera_look;
  float3   camera_up;
  float3   camera_right;
  float    camera_tan_fov_05;
  float    camera_aspect;
};
[[vk::binding(0, 1)]] ConstantBuffer<FrameConstants> fc : register(b0, space1);

[[vk::binding(0, 0)]] RWTexture2D<float4> targets[2] : register(u0, space0);

[[vk::binding(1, 0)]] SamplerState ss : register(s1, space0);

[[vk::binding(2, 0)]] Texture2D<float4> inputs[32] : register(t2, space0);

#define GBUFFER_NORMAL 0
#define GBUFFER_VPOS 1
#define GBUFFER_DEPTH 2
#define GIZMO_LAYER 5
#define DEPTH_PYRAMID_MIP0 16

// o  - ray origin
// id - inversed ray direction

float ray_box(float2 o, float2 id, float xy_plane) {
  float2 t = (xy_plane - o) * id;
  return min(t.x, t.y);
}

// void step_ray(float2 o, float2 id, )

// void gen_camera_ray(in float2 ndc_xy, out float3 ro, out float3 rd) {
//   ro = fc.camera_pos;
//   rd = normalize(fc.camera_look + fc.camera_tan_fov_05 * (fc.camera_right * ndc_xy.x *
//   fc.camera_aspect + fc.camera_up * ndc_xy.y));
//}

// View space ray
void gen_camera_ray(in float2 ndc_xy, out float3 ro, out float3 rd) {
  ro = float3(0.0f, 0.0f, 0.0f);
  rd = normalize(float3(0.0f, 0.0f, 1.0f) +
                 fc.camera_tan_fov_05 * (float3(1.0f, 0.0f, 0.0f) * ndc_xy.x * fc.camera_aspect +
                                         float3(0.0f, 1.0f, 0.0f) * ndc_xy.y));
}

float3 rotate(float4x4 mat, float3 vec) {
  return mul(float3x3(mat[0].xyz, mat[1].xyz, mat[2].xyz), vec);
}

float3 ssr_dummy(float3 ro, float3 rd) {
  float dt = 1.0e-2f;
  uint2 isize;
  float iz = -ro.z;
  inputs[DEPTH_PYRAMID_MIP0].GetDimensions(isize.x, isize.y);
  for (u32 i = 1; i < 256; i++) {
    float3 p  = ro + rd * dt * float(i);
    float4 pp = mul(fc.view_to_proj, float4(p, 1.0f));
    pp.xy /= pp.w;
    if (any(abs(pp.xy) > 1.0f)) return float3(0.0f, 0.0f, 0.0f);
    float2 uv    = pp.xy * 0.5f + 0.5f;
    uv.y         = 1.0f - uv.y;
    int2  ixy    = int2(uv * float2(isize));
    float ldepth = inputs[DEPTH_PYRAMID_MIP0].Load(int3(ixy, 0)).x;
    if (-p.z > ldepth && iz < ldepth) return float3(uv, 1.0f);
  }
  return float3(0.0f, 0.0f, 0.0f);
}

float3 ssr_dda(float3 ro, float3 rd) {
  uint2 isize;
  inputs[DEPTH_PYRAMID_MIP0].GetDimensions(isize.x, isize.y);

  float4 start_p = mul(fc.view_to_proj, float4(ro, 1.0f));
  start_p.xyz    = start_p.xyz / start.w;
  start_p.xy     = start_p.xy * float2(0.5f, -0.5f) + 0.5f;
  start_p.xy *= float2(isize);

  float4 end_p = mul(fc.view_to_proj, float4(ro + rd * 10.0f, 1.0f));
  end_p.xyz    = end_p.xyz / start.w;
  end_p.xy     = end_p.xy * float2(0.5f, -0.5f) + 0.5f;
  end_p.xy *= float2(isize);

  float dt = 1.0e-2f;
  uint2 isize;
  float iz = -ro.z;
  inputs[DEPTH_PYRAMID_MIP0].GetDimensions(isize.x, isize.y);
  float2 dr;
  float2 slope;
  rd = rotate(fc.view_to_proj, rd);
  if (abs(rd.x) > abs(rd.y)) {
    dr    = float2(1.0f * sign(rd.x), 0.0f);
    slope = float2(0.0f, rd.y / rd.x);
  } else {
    dr    = float2(0.0f, 1.0f * sign(rd.y));
    slope = float2(rd.x / rd.y, 0.0f);
  }

  float3 p  = ro;
  float4 pp = mul(fc.view_to_proj, float4(p, 1.0f));
  pp.xy /= pp.w;
  if (any(abs(pp.xy) > 1.0f)) return float3(0.0f, 0.0f, 0.0f);
  float2 uv = pp.xy * 0.5f + 0.5f;
  uv.y      = 1.0f - uv.y;
  int2 ixy  = int2(uv * float2(isize));

  for (u32 i = 0; i < 128; i++) {
    float2 total_dr = dr * i;
    int2   icoord   = ixy + int2(total_dr) + int2((total_dr.x + total_dr.y) * slope);
    // targets[1][icoord] = float4(1.0f, 1.0f, 0.0f, 1.0f);
    float ldepth = inputs[DEPTH_PYRAMID_MIP0].Load(int3(icoord, 0)).x;
  }
  return float3(0.0f, 0.0f, 0.0f);
}

float3 ssr_hiz(float3 ro, float3 rd) {
  float dt = 1.0e-2f;
  uint2 isize;
  float iz        = -ro.z;
  u32   mip_level = 4;
  float dz        = 1.0e-2f;
  inputs[DEPTH_PYRAMID_MIP0].GetDimensions(isize.x, isize.y);
  float2 floor_offset = rd > 0.0f ? 0.0f : 1.0f;
  for (u32 i = 1; i < 256; i++) {
    uint2 level_size = uint2(max(1, isize.x >> mip_level), max(1, isize.y >> mip_level));
    // inputs[DEPTH_PYRAMID_MIP0].GetDimensions(level_size.x, level_size.y);
    float3 p  = ro + rd * dt * float(i);
    float4 pp = mul(fc.view_to_proj, float4(p, 1.0f));
    pp.xy /= pp.w;
    if (any(abs(pp.xy) > 1.0f)) return float3(0.0f, 0.0f, 0.0f);
    float2 uv    = pp.xy * 0.5f + 0.5f;
    uv.y         = 1.0f - uv.y;
    int2  ixy    = int2(uv * float2(level_size));
    float ldepth = inputs[DEPTH_PYRAMID_MIP0].Load(int3(ixy, mip_level)).x;
    if (-p.z > ldepth + dz && iz < ldepth - dz) return float3(uv, 1.0f);
  }
  return float3(0.0f, 0.0f, 0.0f);
}

[numthreads(16, 16, 1)] void main(uint3 tid
                                  : SV_DispatchThreadID) {
  uint width, height;
  targets[0].GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height) return;
  float2 uv     = (float2(tid.xy) + float2(0.5f, 0.5f)) / float2(width, height);
  float3 normal = inputs[GBUFFER_NORMAL].Load(int3(tid.xy, 0)).xyz;
  float3 vpos   = inputs[GBUFFER_VPOS].Load(int3(tid.xy, 0)).xyz;
  float  depth  = inputs[GBUFFER_DEPTH].Load(int3(tid.xy, 0)).x;
  float  ldepth = inputs[DEPTH_PYRAMID_MIP0 + 0].Load(int3(tid.xy, 0)).x;
  float4 gizmo  = inputs[GIZMO_LAYER].Load(int3(tid.xy, 0)).xyzw;
  float3 color  = float3_splat(
      max(0.0f, dot(rotate(fc.view_to_world, normal), normalize(float3(1.0, 1.0, 1.0)))));
  targets[0][tid.xy] = float4(lerp(pow(color, 1.0f), gizmo.xyz, gizmo.w), 1.0f);
  float2 ndc_xy      = uv * 2.0f - float2_splat(1.0f);
  ndc_xy.y *= -1.0f;
  float3 ro, rd;
  gen_camera_ray(ndc_xy, ro, rd);
  float3 prim_col = rd * ldepth / rd.z;
  prim_col.z *= -1.0f;
  rd.z *= -1.0f;
  // targets[1][tid.xy] = float4(abs(reflect(rd, normal)), 1.0f);
  // targets[1][tid.xy] = float4(abs(mul(fc.view_to_world, float4(prim_col.xyz, 1.0f))).xyz, 1.0f);
  // if (ldepth > 1.0e3f) {
  //  targets[1][tid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
  //  return;
  //}
  /* else
    targets[1][tid.xy] = float4(ssr_hiz(prim_col, reflect(rd, normal)), 1.0f);*/
  // targets[1][tid.xy] = float4(mul(fc.view_to_world, float4(prim_col.xyz, 1.0f)).xyz, 1.0f);
  // targets[1][tid.xy] = float4(abs(vpos.xyz - prim_col.xyz) * 1.e4f, 1.0f);
  // targets[1][tid.xy] = float4(prim_col.xyz, 1.0f);
  // if (length(ndc_xy) < 0.01f)
  ssr_dda(prim_col, reflect(rd, normal));
}