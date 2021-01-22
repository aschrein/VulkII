#include "examples/shaders/declarations.hlsl"

[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

[[vk::binding(0, 1)]] ConstantBuffer<FrameConstants> fc : register(b0, space1);

[[vk::binding(0, 2)]] RWByteAddressBuffer feedback_buffer : register(u0, space2);

[[vk::binding(0, 0)]] RWByteAddressBuffer raw_attributes : register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer raw_indices : register(u1, space0);
[[vk::binding(2, 0)]] RWTexture2D<float4> normal_target : register(u2, space0);
[[vk::binding(3, 0)]] RWTexture2D<uint>   depth_target : register(u3, space0);

u32 fetch_index(u32 index_id) {
  if (pc.index_stride == 4) {
    return raw_indices.Load<uint>(pc.index_offset + index_id * 4);
  } else if (pc.index_stride == 2) {
    u32 uint_id = (index_id + pc.index_offset) / 2;
    u32 dword   = raw_indices.Load<uint>(uint_id * 4);
    u32 word_id = (index_id + pc.index_offset) & 1u;
    return ((dword >> (word_id * 16u)) & 0xffffu);
  } else {
    return 0;
  }
}

float3 fetch_position(u32 vertex_id) {
  return raw_attributes.Load<float3>(pc.position_offset + pc.position_stride * vertex_id);
}

float3 fetch_normal(u32 vertex_id) {
  return raw_attributes.Load<float3>(pc.normal_offset + pc.normal_stride * vertex_id);
}

bool in_bounds(float4 p) { return p.w > 0.0 && p.z > 0.0 && abs(p.x) < p.w && abs(p.y) < p.w; }

bool outside(int2 e1, int2 e2) { return e2.x * e1.y - e2.y * e1.x < 0; }

[numthreads(RASTERIZATION_GROUP_SIZE, 1, 1)] void main(uint3 DTid
                                                       : SV_DispatchThreadID) {
  uint width, height;
  normal_target.GetDimensions(width, height);
  if (DTid.x > pc.index_count / 3) return;
  u32 id = DTid.x;
  u32 i0 = fetch_index(id * 3 + 0);
  u32 i1 = fetch_index(id * 3 + 1);
  u32 i2 = fetch_index(id * 3 + 2);

  float3 p0 = fetch_position(pc.first_vertex + i0);
  float3 p1 = fetch_position(pc.first_vertex + i1);
  float3 p2 = fetch_position(pc.first_vertex + i2);

  // feedback_buffer.Store<float3>((DTid.x * 3 + 0) * 12, p0);
  // feedback_buffer.Store<float3>((DTid.x * 3 + 1) * 12, p1);
  // feedback_buffer.Store<float3>((DTid.x * 3 + 2) * 12, p2);

  float4 pp0 = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
  float4 pp1 = mul(fc.viewproj, mul(pc.model, float4(p1, 1.0)));
  float4 pp2 = mul(fc.viewproj, mul(pc.model, float4(p2, 1.0)));

  if (!in_bounds(pp0) || !in_bounds(pp1) || !in_bounds(pp2)) return;

  pp0 /= pp0.w;
  pp1 /= pp1.w;
  pp2 /= pp2.w;
  //
  // For simplicity, we assume samples are at pixel centers
  //  __________
  // |          |
  // |          |
  // |    X     |
  // |          |
  // |__________|
  //

  // v_i - Vertices scaled to window size so 1.5 is inside the second pixel
  float2 v0 = float2(float(width) * (pp0.x + 1.0) / 2.0, float(height) * (-pp0.y + 1.0) / 2.0);
  float2 v1 = float2(float(width) * (pp1.x + 1.0) / 2.0, float(height) * (-pp1.y + 1.0) / 2.0);
  float2 v2 = float2(float(width) * (pp2.x + 1.0) / 2.0, float(height) * (-pp2.y + 1.0) / 2.0);

  // Edges
  float2 n0 = v1 - v0;
  float2 n1 = v2 - v1;
  float2 n2 = v0 - v2;

  // Double area
  float area2 = (n0.x * n2.y - n0.y * n2.x);

  // Back/small triangle culling
  if (area2 < 1.0e-6f) return;

  // 2D Edge Normals
  n0 = -float2(-n0.y, n0.x) / area2;
  n1 = -float2(-n1.y, n1.x) / area2;
  n2 = -float2(-n2.y, n2.x) / area2;

  // Bounding Box
  float2 fmin = float2(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)));
  float2 fmax = float2(max(v0.x, max(v1.x, v2.x)), max(v0.y, max(v1.y, v2.y)));

  int2 imin = int2(fmin);
  int2 imax = int2(fmax);

  // Edge function values at the first (imin.x + 0.5f, imin.y + 0.5f) sample position
  float2 first_sample = float2(imin) + float2(0.5f, 0.5f);
  float  init_ef0     = dot(first_sample - v0, n0);
  float  init_ef1     = dot(first_sample - v1, n1);
  float  init_ef2     = dot(first_sample - v2, n2);

  float3 normal_0 = fetch_normal(pc.first_vertex + i0);
  float3 normal_1 = fetch_normal(pc.first_vertex + i1);
  float3 normal_2 = fetch_normal(pc.first_vertex + i2);

  // Still a scalar loop
  //[unroll(2)]
  for (i32 dy = 0; dy <= imax.y - imin.y; dy += 1) {
    //[unroll(2)]
    for (i32 dx = 0; dx <= imax.x - imin.x; dx += 1) {

      i32 x = imin.x + dx;
      i32 y = imin.y + dy;

      if (x >= 0 && y >= 0 && x < height && y < height) {
        float ef0 = init_ef0 + n0.x * float(dx) + n0.y * float(dy);
        float ef1 = init_ef1 + n1.x * float(dx) + n1.y * float(dy);
        float ef2 = init_ef2 + n2.x * float(dx) + n2.y * float(dy);

        if (ef0 < 0.0f || ef1 < 0.0f || ef2 < 0.0f) continue;

        // Barycentrics
        float b0 = ef1;
        float b1 = ef2;
        float b2 = ef0;
        // Perspective correction
        float bw = b0 / pp0.w + b1 / pp1.w + b2 / pp2.w;
        b0       = b0 / pp0.w / bw;
        b1       = b1 / pp1.w / bw;
        b2       = b2 / pp2.w / bw;

        // add tid.x * 1.0e-6f to avoid z-fight
        u32 depth = u32((pp0.z) * 1000000);
        u32 next_depth;
        InterlockedMax(depth_target[int2(x, y)], depth, next_depth);
        if (depth > next_depth) {
          float3 pixel_normal       = normalize(normal_0 * b0 + normal_1 * b1 + normal_2 * b2);
          normal_target[int2(x, y)] = float4(
              float3_splat(max(0.0f, dot(pixel_normal, normalize(float3_splat(1.0f))))), 1.0f);
        }
      }
    }
  }
}