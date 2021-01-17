#include "examples/shaders/declarations.hlsl"

[[vk::push_constant]] ConstantBuffer<GI_PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

[[vk::binding(0, 1)]] ConstantBuffer<FrameConstants> fc : register(b0, space1);

// Buffer for debugging
// [[vk::binding(0, 2)]] RWByteAddressBuffer feedback_buffer : register(u0, space2);

// Source Buffers
[[vk::binding(0, 0)]] Texture2D<float4> position_source : register(t0, space0);
[[vk::binding(1, 0)]] Texture2D<float4> normal_source : register(t1, space0);
// Target Buffers
[[vk::binding(2, 0)]] RWTexture2D<float4> normal_target : register(u2, space0);
[[vk::binding(3, 0)]] RWTexture2D<uint>   depth_target : register(u3, space0);
// Atomic counter for max pixels/triangle per patch
[[vk::binding(4, 0)]] RWTexture2D<uint> counter_grid : register(u4, space0);

[[vk::binding(5, 0)]] SamplerState ss : register(s5, space0);

//
[[vk::binding(6, 0)]] RWTexture2D<uint> grid_resolution_table : register(u6, space0);
[[vk::binding(7, 0)]] RWTexture2D<uint> subgrid_counter_table : register(u7, space0);

//[[vk::binding(7, 0)]] RWTexture2D<uint>   prev_counter_grid : register(u7, space0);

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
  uv           = frac(uv * 15.718281828459045);
  float3 seeds = float3(0.123, 0.456, 0.789);
  seeds        = frac((uv.x + 0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.x, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  seeds        = frac((uv.y + 0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.y, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  seeds        = frac((0.5718281828459045 + seeds) *
               ((seeds + fmod(uv.x, 0.141592653589793)) * 27.61803398875 + 4.718281828459045));
  return seeds;
}

// groupshared u32 num_samples;
// uint3 group_id : SV_GroupID
// Spawn the whole wavefront at a time
[numthreads(64, 1, 1)] void main(uint3 group_id : SV_GroupID) {
  // Target resolution
  uint width, height;
  normal_target.GetDimensions(width, height);

  // Resolution of the source images
  u32 src_res;
  {
    uint width, height;
    normal_source.GetDimensions(width, height);
    src_res = width;
  }

  u32 cell_x = (group_id % COUNTER_GRID_RESOLUTION);
  u32 cell_y = (group_id / COUNTER_GRID_RESOLUTION);

  // Scalar loops
  //[unroll]
  //for (u32 cell_y = 0; cell_y < COUNTER_GRID_RESOLUTION; cell_y++)
  {
    //[unroll]
    //for (u32 cell_x = 0; cell_x < COUNTER_GRID_RESOLUTION; cell_x++)
    {
      u32 cell_res = grid_resolution_table[int2(cell_x, cell_y)];
      if (cell_res == 0) return;
      float2 cell_uv = float2(cell_x, cell_y) / float(COUNTER_GRID_RESOLUTION);
      while (true) {
        u32 subcell_offset;
        if (WaveGetLaneIndex() == 0) {
          /*         subgrid_counter_table.InterlockedAdd(4 * (cell_x + cell_y *
             COUNTER_GRID_RESOLUTION), 1, subcell_offset);*/
          InterlockedAdd(subgrid_counter_table[int2(cell_x, cell_y)], 1, subcell_offset);
        }
        subcell_offset = WaveReadLaneAt(subcell_offset, 0);
        // All subcells are already finished
        if (subcell_offset >= cell_res * cell_res) return;
        u32    subcell_x       = (subcell_offset % cell_res);
        u32    subcell_y       = (subcell_offset / cell_res);
        f32    subcell_uv_step = 1.0f / float(cell_res * COUNTER_GRID_RESOLUTION);
        float2 subcell_uv      = cell_uv + float2(subcell_x, subcell_y) * subcell_uv_step;
        int2   offsets[6]      = {
            int2(0, 0), int2(0, 1), int2(1, 0), int2(1, 0), int2(0, 1), int2(1, 1),
        };

        f32 mip_level = log2(subcell_uv_step * f32(src_res));

        // u32 tri_masks[6] = {0, 2, 1, 1, 2, 3};

        // Scalar loop
        [unroll] for (u32 tri_id = 0; tri_id < 2; tri_id++) {
          /*float2 suv0 = subcell_uv + float2((tri_masks[0 + tri_id * 3]) & 1,
                                            (tri_masks[0 + tri_id * 3] >> 1) & 1) *
                                         subcell_uv_step;
          float2 suv1 = subcell_uv + float2((tri_masks[1 + tri_id * 3]) & 1,
                                            (tri_masks[1 + tri_id * 3] >> 1) & 1) *
                                         subcell_uv_step;
          float2 suv2 = subcell_uv + float2((tri_masks[2 + tri_id * 3]) & 1,
                                            (tri_masks[2 + tri_id * 3] >> 1) & 1) *
                                         subcell_uv_step;*/
          float2 suv0 = subcell_uv + offsets[0 + tri_id * 3] * subcell_uv_step;
          float2 suv1 = subcell_uv + offsets[1 + tri_id * 3] * subcell_uv_step;
          float2 suv2 = subcell_uv + offsets[2 + tri_id * 3] * subcell_uv_step;
          //if (WaveGetLaneIndex() == 0) add_sample((suv0 + suv1 + suv2) / 3.0f, 1);
          //continue;
#if 1
          // float2 suv0 = subcell_uv + float2(0.0f, 0.0f);
          // float2 suv1 = subcell_uv + float2(subcell_uv_step, 0.0f);
          // float2 suv2 = subcell_uv + float2(0.0f, subcell_uv_step);
          float4 pp0 = position_source.SampleLevel(ss, suv0, mip_level).xyzw;
          float4 pp1 = position_source.SampleLevel(ss, suv1, mip_level).xyzw;
          float4 pp2 = position_source.SampleLevel(ss, suv2, mip_level).xyzw;
          pp0        = mul(fc.viewproj, mul(pc.model, float4(pp0.xyz, 1.0)));
          pp1        = mul(fc.viewproj, mul(pc.model, float4(pp1.xyz, 1.0)));
          pp2        = mul(fc.viewproj, mul(pc.model, float4(pp2.xyz, 1.0)));
          // For simplicity just discard triangles that touch the boundary
          // @TODO(aschrein): Add proper clipping.
          if (!in_bounds(pp0) || !in_bounds(pp1) || !in_bounds(pp2)) continue;

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
          float2 v0 =
              float2(float(width) * (pp0.x + 1.0) / 2.0, float(height) * (-pp0.y + 1.0) / 2.0);
          float2 v1 =
              float2(float(width) * (pp1.x + 1.0) / 2.0, float(height) * (-pp1.y + 1.0) / 2.0);
          float2 v2 =
              float2(float(width) * (pp2.x + 1.0) / 2.0, float(height) * (-pp2.y + 1.0) / 2.0);

          // Edges
          float2 n0 = v1 - v0;
          float2 n1 = v2 - v1;
          float2 n2 = v0 - v2;

          // Double area
          float area2 = (n0.x * n2.y - n0.y * n2.x);

          // Back/small triangle culling
          if (area2 < 1.0e-6f) continue;

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

          // if (WaveGetLaneIndex() == 0) num_samples = 0;
          volatile u32 num_samples = 0;

          // Bound the maximum triangle size to 32x32 pixels
          imax.x = imin.x + min(32, imax.x - imin.x);
          imax.y = imin.y + min(32, imax.y - imin.y);

          // Still a scalar loop
          [unroll(4)] for (i32 dy = 0; dy <= imax.y - imin.y; dy += 8) {
            [unroll(4)] for (i32 dx = 0; dx <= imax.x - imin.x; dx += 8) {
              // Per lane iteration starts here
              // 64 lanes or 8x8 lanes
              u32 lane_x = WaveGetLaneIndex() & 0x7;
              u32 lane_y = WaveGetLaneIndex() >> 3;
              i32 x      = imin.x + dx + lane_x;
              i32 y      = imin.y + dy + lane_y;
              // normal_target[int2(x, y)] = float4(1.0f, 0.0f, 0.0f, 1.0f);

              if (x >= 0 && y >= 0 && x < height && y < height) {
                float ef0 = init_ef0 + n0.x * float(dx + lane_x) + n0.y * float(dy + lane_y);
                float ef1 = init_ef1 + n1.x * float(dx + lane_x) + n1.y * float(dy + lane_y);
                float ef2 = init_ef2 + n2.x * float(dx + lane_x) + n2.y * float(dy + lane_y);

                if (ef0 < 0.0f || ef1 < 0.0f || ef2 < 0.0f) continue;

#  if 0
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
                //float2 pixel_uv = suv0 * b0 + suv1 * b1 + suv2 * b2;

                // add tid.x * 1.0e-6f to avoid z-fight
                u32 depth = u32((pp0.z) * 1000000);
                u32 next_depth;
                InterlockedMax(depth_target[int2(x, y)], depth, next_depth);
                if (depth > next_depth) {
                  num_samples++;
                  // InterlockedAdd(num_samples, 1);
                  if (pc.flags & GI_RASTERIZATION_FLAG_PIXEL_COLOR_TRIANGLES)
                    normal_target[int2(x, y)] =
                        // float4(random_color(suv0) * random_color(uv_offset), 1.0f);
                        float4(float2_splat(mip_level / 10.0f), random_color(suv0).z, 1.0f);
                  else {
                    float3 normal_0     = normal_source.SampleLevel(ss, suv0, mip_level).xyz;
                    float3 normal_1     = normal_source.SampleLevel(ss, suv1, mip_level).xyz;
                    float3 normal_2     = normal_source.SampleLevel(ss, suv2, mip_level).xyz;
                    float3 pixel_normal = normalize(normal_0 * b0 + normal_1 * b1 + normal_2 * b2);
                    normal_target[int2(x, y)] = float4(
                        float3_splat(max(0.0f, dot(pixel_normal, normalize(float3_splat(1.0f))))),
                        1.0f);
                  }
                }
#  else
                u32 depth = u32((pp0.z) * 1000000);
                u32 next_depth;
                InterlockedMax(depth_target[int2(x, y)], depth, next_depth);
                if (depth > next_depth) {
                  num_samples++;
                  normal_target[int2(x, y)] =
                      float4(float2_splat(mip_level / 10.0f), random_color(suv0).z, 1.0f);
                }
#  endif
              }
            }
          }
          /* [unroll] for (u32 i = 32; i > 0; i = i >> 1) {
   num_samples += WaveReadLaneAt(WaveGetLaneIndex() + i, num_samples);
 }*/
          /*num_samples += WaveReadLaneAt(WaveGetLaneIndex() + 32, num_samples);
          num_samples += WaveReadLaneAt(WaveGetLaneIndex() + 16, num_samples);
          num_samples += WaveReadLaneAt(WaveGetLaneIndex() + 8, num_samples);
          num_samples += WaveReadLaneAt(WaveGetLaneIndex() + 4, num_samples);
          num_samples += WaveReadLaneAt(WaveGetLaneIndex() + 2, num_samples);
          num_samples += WaveReadLaneAt(WaveGetLaneIndex() + 1, num_samples);*/
          // num_samples = WavePrefixSum(num_samples) + num_samples;
          num_samples = WaveActiveSum(num_samples);
          // the first lane adds samples for the whole wavefront
          if (WaveGetLaneIndex() == 0) add_sample((suv0 + suv1 + suv2) / 3.0f, num_samples);
            // return;
#endif
        }
      }
    }
  }
#if 0
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
  int2   offsets[6] = {
      int2(0, 0), int2(0, 1), int2(1, 0), int2(1, 0), int2(0, 1), int2(1, 1),
  };
  // Compute the needed LOD for reading
  u32 src_res;
  {
    uint width, height;
    normal_source.GetDimensions(width, height);
    src_res = width;
  }
  f32 pixels_covered = uv_step * f32(src_res);
  f32 mip_level      = log2(pixels_covered);

  {
    float2 suv0 = grid_uv0;
    float3 p0   = position_source.SampleLevel(ss, suv0, mip_level).xyz;
    float4 pp0  = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
    vertex_cache[gid.x + gid.y * (GI_RASTERIZATION_GROUP_SIZE + 1)] = pp0;
  }
  if (gid.x == GI_RASTERIZATION_GROUP_SIZE - 1) {
    float2 suv0 = grid_uv0 + float2(uv_step.x, 0.0f);
    float3 p0   = position_source.SampleLevel(ss, suv0, mip_level).xyz;
    float4 pp0  = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
    vertex_cache[gid.x + 1 + gid.y * (GI_RASTERIZATION_GROUP_SIZE + 1)] = pp0;
  }
  if (gid.y == GI_RASTERIZATION_GROUP_SIZE - 1) {
    float2 suv0 = grid_uv0 + float2(0.0f, uv_step.x);
    float3 p0   = position_source.SampleLevel(ss, suv0, mip_level).xyz;
    float4 pp0  = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
    vertex_cache[gid.x + (gid.y + 1) * (GI_RASTERIZATION_GROUP_SIZE + 1)] = pp0;
  }
  if (gid.y == GI_RASTERIZATION_GROUP_SIZE - 1 && gid.x == GI_RASTERIZATION_GROUP_SIZE - 1) {
    float2 suv0 = grid_uv0 + float2(uv_step.x, uv_step.x);
    float3 p0   = position_source.SampleLevel(ss, suv0, mip_level).xyz;
    float4 pp0  = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
    vertex_cache[gid.x + 1 + (gid.y + 1) * (GI_RASTERIZATION_GROUP_SIZE + 1)] = pp0;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint tri_id = 0; tri_id < 2; tri_id++) {
    float2 suv0 = grid_uv0 + offsets[0 + tri_id * 3] * uv_step;
    float2 suv1 = grid_uv0 + offsets[1 + tri_id * 3] * uv_step;
    float2 suv2 = grid_uv0 + offsets[2 + tri_id * 3] * uv_step;

    /*float3 p0 = position_source.SampleLevel(ss, suv0, mip_level).xyz;
    float3 p1 = position_source.SampleLevel(ss, suv1, mip_level).xyz;
    float3 p2 = position_source.SampleLevel(ss, suv2, mip_level).xyz;*/

    /*float alpha0 = position_source.SampleLevel(ss, suv0, 0.0f).w;
    float alpha1 = position_source.SampleLevel(ss, suv1, 0.0f).w;
    float alpha2 = position_source.SampleLevel(ss, suv2, 0.0f).w;
    if (alpha0 < 0.5f || alpha1 < 0.5f || alpha2 < 0.5f) return;*/

    // feedback_buffer.Store<float3>((DTid.x * 3 + 0) * 12, p0);
    // feedback_buffer.Store<float3>((DTid.x * 3 + 1) * 12, p1);
    // feedback_buffer.Store<float3>((DTid.x * 3 + 2) * 12, p2);

    // Screen space projected positions
    /*float4 pp0 = mul(fc.viewproj, mul(pc.model, float4(p0, 1.0)));
    float4 pp1 = mul(fc.viewproj, mul(pc.model, float4(p1, 1.0)));
    float4 pp2 = mul(fc.viewproj, mul(pc.model, float4(p2, 1.0)));*/

    float4 pp0 =
        vertex_cache[(gid.x + offsets[0 + tri_id * 3].x) +
                     (gid.y + offsets[0 + tri_id * 3].y) * (GI_RASTERIZATION_GROUP_SIZE + 1)];
    float4 pp1 =
        vertex_cache[(gid.x + offsets[1 + tri_id * 3].x) +
                     (gid.y + offsets[1 + tri_id * 3].y) * (GI_RASTERIZATION_GROUP_SIZE + 1)];
    float4 pp2 =
        vertex_cache[(gid.x + offsets[2 + tri_id * 3].x) +
                     (gid.y + offsets[2 + tri_id * 3].y) * (GI_RASTERIZATION_GROUP_SIZE + 1)];

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
    float area2 = (e0.x * e2.y - e0.y * e2.x);

    // Back/small triangle culling
    if (area2 < 1.0e-6f) return;

    // 2D Edge Normals
    float2 n0 = -float2(-e0.y, e0.x) / area2;
    float2 n1 = -float2(-e1.y, e1.x) / area2;
    float2 n2 = -float2(-e2.y, e2.x) / area2;

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

    u32 num_samples = 0;

    float3 normal_0 = normal_source.SampleLevel(ss, suv0, mip_level).xyz;
    float3 normal_1 = normal_source.SampleLevel(ss, suv1, mip_level).xyz;
    float3 normal_2 = normal_source.SampleLevel(ss, suv2, mip_level).xyz;

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
              normal_target[int2(x, y)] =
                  // float4(random_color(suv0) * random_color(uv_offset), 1.0f);
                  float4(float2_splat(mip_level / 10.0f), random_color(suv0).z, 1.0f);
            else
              normal_target[int2(x, y)] = float4(
                  float3_splat(max(0.0f, dot(pixel_normal, normalize(float3_splat(1.0f))))), 1.0f);
          }
        }
      }
    }
    add_sample((suv0 + suv1 + suv2) / 3.0f, num_samples);
  }
#endif
}