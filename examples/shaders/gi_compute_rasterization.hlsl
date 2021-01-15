#include "examples/shaders/declarations.hlsl"

[[vk::push_constant]] ConstantBuffer<GI_PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

[[vk::binding(0, 1)]] ConstantBuffer<FrameConstants> fc : register(b0, space1);

// Buffer for debugging
// [[vk::binding(0, 2)]] RWByteAddressBuffer feedback_buffer : register(u0, space2);

[[vk::binding(0, 0)]] Texture2D<float4> position_source : register(t0, space0);
[[vk::binding(1, 0)]] Texture2D<float4> normal_source : register(t1, space0);

[[vk::binding(2, 0)]] RWTexture2D<float4> normal_target : register(u2, space0);
[[vk::binding(3, 0)]] RWTexture2D<uint>   depth_target : register(u3, space0);
[[vk::binding(4, 0)]] RWTexture2D<uint>   counter_grid : register(u4, space0);
[[vk::binding(5, 0)]] SamplerState        ss : register(s5, space0);
[[vk::binding(6, 0)]] RWByteAddressBuffer indirect_arg_buffer : register(u6, space0);
[[vk::binding(7, 0)]] RWTexture2D<uint>   prev_counter_grid : register(u7, space0);

struct IndirectArgs {
  u32 dimx;
  u32 dimy;
  u32 dimz;
};

bool in_bounds(float4 p) { return p.w > 0.0 && p.z > 0.0 && abs(p.x) < p.w && abs(p.y) < p.w; }

bool outside(int2 e1, int2 e2) { return e2.x * e1.y - e2.y * e1.x < 0; }
bool outside(float2 e1, float2 e2) { return e2.x * e1.y - e2.y * e1.x < 0.0f; }

void add_sample(float2 uv, u32 num) {
  u32 width, height;
  counter_grid.GetDimensions(width, height);
  u32 ix = uv.x * width;
  u32 iy = uv.y * height;
  InterlockedMax(counter_grid[int2(ix, iy)], num);
}

float3 random_color(float2 uv) {
  uv           = uv * 15.718281828459045;
  float3 seeds = float3(0.123, 0.456, 0.789);
  seeds        = frac((uv.x + 0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.x, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  seeds        = frac((uv.y + 0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.y, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  seeds        = frac((0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.x, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  return seeds;
}

[numthreads(GI_RASTERIZATION_GROUP_SIZE, GI_RASTERIZATION_GROUP_SIZE, 1)] void
main(uint3 tid
     : SV_DispatchThreadID) {
  float2 uv_offset = float2(0.0f, 0.0f);
  float2 uv_step   = float2(1.0f, 1.0f) / float(COUNTER_GRID_RESOLUTION);
  {
    IndirectArgs args = indirect_arg_buffer.Load<IndirectArgs>(
        12 * (pc.cell_x + pc.cell_y * COUNTER_GRID_RESOLUTION));
    u32 res   = GI_RASTERIZATION_GROUP_SIZE * args.dimx;
    uv_offset = float2(float(pc.cell_x), float(pc.cell_y)) /
                float2(float(COUNTER_GRID_RESOLUTION), float(COUNTER_GRID_RESOLUTION));
    float cell_size = 1.0f / float(COUNTER_GRID_RESOLUTION);
    uv_step         = cell_size / float(res);
  }

  uint width, height;
  normal_target.GetDimensions(width, height);
  uint2  pnt        = tid.xy;
  float2 grid_uv0   = float2(pnt) * uv_step + uv_offset;
  float2 offsets[6] = {
      float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f),
      float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f),
  };
  for (uint tri_id = 0; tri_id < 2; tri_id++) {
    float2 suv0 = grid_uv0 + offsets[0 + tri_id * 3] * uv_step;
    float2 suv1 = grid_uv0 + offsets[1 + tri_id * 3] * uv_step;
    float2 suv2 = grid_uv0 + offsets[2 + tri_id * 3] * uv_step;
    float3 p0   = position_source.SampleLevel(ss, suv0, 0.0f).xyz;
    float3 p1   = position_source.SampleLevel(ss, suv1, 0.0f).xyz;
    float3 p2   = position_source.SampleLevel(ss, suv2, 0.0f).xyz;

    float3 normal_0 = normal_source.SampleLevel(ss, suv0, 0.0f).xyz;
    float3 normal_1 = normal_source.SampleLevel(ss, suv1, 0.0f).xyz;
    float3 normal_2 = normal_source.SampleLevel(ss, suv2, 0.0f).xyz;

    // feedback_buffer.Store<float3>((DTid.x * 3 + 0) * 12, p0);
    // feedback_buffer.Store<float3>((DTid.x * 3 + 1) * 12, p1);
    // feedback_buffer.Store<float3>((DTid.x * 3 + 2) * 12, p2);

    // Screen space projected positions
    float4 pp0 = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
    float4 pp1 = mul(fc.viewproj, mul(pc.model, float4(p1, 1.0)));
    float4 pp2 = mul(fc.viewproj, mul(pc.model, float4(p2, 1.0)));

    // For simplicity just discard triangles that touch the boundary
    // @TODO(aschrein): Add proper clipping.
    if (!in_bounds(pp0) || !in_bounds(pp1) || !in_bounds(pp2)) return;

    pp0.xyz /= pp0.w;
    pp1.xyz /= pp1.w;
    pp2.xyz /= pp2.w;

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
    float2 e0 = v1 - v0;
    float2 e1 = v2 - v1;
    float2 e2 = v0 - v2;

    // Double area
    float area2 = e0.x * e2.y - e0.y * e2.x;

    // Back/small triangle culling
    if (area2 < 1.0e-6f) return;

    // 2D Edge Normals
    float2 n0 = -float2(-e0.y, e0.x) / area2;
    float2 n1 = -float2(-e1.y, e1.x) / area2;
    float2 n2 = -float2(-e2.y, e2.x) / area2;

    // Bounding Box
    float2 fmin = float2(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)));
    float2 fmax = float2(max(v0.x, max(v1.x, v2.x)), max(v0.y, max(v1.y, v2.y)));

    int2 ip0 = int2(v0);
    int2 ip1 = int2(v1);
    int2 ip2 = int2(v2);

    int2 imin = int2(fmin);
    int2 imax = int2(fmax);

    // Edge function values at the first (imin.x + 0.5f, imin.y + 0.5f) sample position
    float2 first_sample = float2(imin) + float2(0.5f, 0.5f);
    float  init_ef0     = dot(first_sample - v0, n0);
    float  init_ef1     = dot(first_sample - v1, n1);
    float  init_ef2     = dot(first_sample - v2, n2);

    u32 num_samples = 0;

    for (i32 dy = 0; dy <= imax.y - imin.y; dy++) {
      for (i32 dx = 0; dx <= imax.x - imin.x; dx++) {
        i32 x = imin.x + dx;
        i32 y = imin.y + dy;
        if (x >= 0 && y >= 0 && x < height && y < height) {
          int2 v = int2(x, y);

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

          // Per pixel Attributes
          float2 pixel_uv     = suv0 * b0 + suv1 * b1 + suv2 * b2;
          float3 pixel_normal = normalize(normal_0 * b0 + normal_1 * b1 + normal_2 * b2);

          // add tid.x * 1.0e-6f to avoid z-fight
          u32 depth = u32((pp0.z + tid.x * 1.0e-6f) * 1000000);
          u32 next_depth;
          InterlockedMax(depth_target[int2(x, y)], depth, next_depth);
          if (depth > next_depth) {
            num_samples++;
            if (pc.flags & GI_RASTERIZATION_FLAG_PIXEL_COLOR_TRIANGLES)
              normal_target[int2(x, y)] = float4(random_color(suv0), 1.0f);
            else
              normal_target[int2(x, y)] = float4(pixel_normal, 1.0f);
          }
        }
      }
    }
    add_sample((suv0 + suv1 + suv2) / 3.0f, num_samples);
  }
}