struct PushConstants {
  float4x4 model;
  u32      normal_offset;
  u32      normal_stride;
  u32      position_offset;
  u32      position_stride;
  u32      first_vertex;
  u32      index_offset;
  u32      index_count;
  u32      index_stride;
  u32      flags;
};

#define RASTERIZATION_FLAG_CULL_PIXELS 0x1
#define RASTERIZATION_GROUP_SIZE 64

struct FrameConstants {
  float4x4 viewproj;
};

#ifdef HLSL

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

  float3 n0 = fetch_normal(pc.first_vertex + i0);
  float3 n1 = fetch_normal(pc.first_vertex + i1);
  float3 n2 = fetch_normal(pc.first_vertex + i2);

  float4 pp0 = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
  float4 pp1 = mul(fc.viewproj, mul(pc.model, float4(p1, 1.0)));
  float4 pp2 = mul(fc.viewproj, mul(pc.model, float4(p2, 1.0)));

  if (!in_bounds(pp0) || !in_bounds(pp1) || !in_bounds(pp2)) return;

  pp0 /= pp0.w;
  pp1 /= pp1.w;
  pp2 /= pp2.w;

  int2 ip0 = int2(i32(0.5 + float(width) * (pp0.x + 1.0) / 2.0),
                  i32(0.5 + float(height) * (-pp0.y + 1.0) / 2.0));
  int2 ip1 = int2(i32(0.5 + float(width) * (pp1.x + 1.0) / 2.0),
                  i32(0.5 + float(height) * (-pp1.y + 1.0) / 2.0));
  int2 ip2 = int2(i32(0.5 + float(width) * (pp2.x + 1.0) / 2.0),
                  i32(0.5 + float(height) * (-pp2.y + 1.0) / 2.0));

  int2 imin = int2(min(ip0.x, min(ip1.x, ip2.x)), min(ip0.y, min(ip1.y, ip2.y)));
  int2 imax = int2(max(ip0.x, max(ip1.x, ip2.x)), max(ip0.y, max(ip1.y, ip2.y)));

  int2 e0 = ip1 - ip0;
  int2 e1 = ip2 - ip1;
  int2 e2 = ip0 - ip2;

  if (outside(e0, e1)) return;

  for (i32 y = imin.y; y <= imax.y; y++) {
    for (i32 x = imin.x; x <= imax.x; x++) {
      if (x >= 0 && y >= 0 && x < height && y < height) {
        int2 v = int2(x, y);
        if (pc.flags & RASTERIZATION_FLAG_CULL_PIXELS) {
          if (outside(e0, v - ip0) || outside(e1, v - ip1) || outside(e2, v - ip2)) continue;
        }
        u32 depth = u32(pp0.z * 1000000);
        u32 next_depth;
        InterlockedMax(depth_target[int2(x, y)], depth, next_depth);
        if (depth > next_depth) {
          normal_target[int2(x, y)] = float4((0.5 * n0 + float3_splat(0.5)), 1.0);
        }
      }
    }
  }

  //
  // i32 x = i32(0.5 + float(width) * (pp0.x + 1.0) / 2.0);
  // i32 y = i32(0.5 + float(height) * (-pp0.y + 1.0) / 2.0);
}
#endif