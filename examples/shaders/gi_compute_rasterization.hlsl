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
  // InterlockedAdd(counter_grid[int2(ix, iy)], num);
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

// Spawn the whole wavefront at a time
[numthreads(64, 1, 1)] void main_wave_raster(uint3 group_id
                                             : SV_GroupID) {
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

  u32 cell_x = pc.cell_x;
  u32 cell_y = pc.cell_y;

  u32 cell_res;
  {
    IndirectArgs args = indirect_arg_buffer.Load<IndirectArgs>(
        12 * (pc.cell_x + pc.cell_y * COUNTER_GRID_RESOLUTION));
    cell_res = args.dimx;
  }
  if (cell_res == 0) return;
  float2 cell_uv = float2(cell_x, cell_y) / float(COUNTER_GRID_RESOLUTION);

  u32    subcell_x       = group_id.x;
  u32    subcell_y       = group_id.y;
  f32    subcell_uv_step = 1.0f / float(cell_res * COUNTER_GRID_RESOLUTION);
  float2 subcell_uv      = cell_uv + float2(subcell_x, subcell_y) * subcell_uv_step;

#ifdef GI_ORDER_CW
  int2 offsets[6] = {
      int2(0, 0), int2(0, 1), int2(1, 0), int2(1, 0), int2(0, 1), int2(1, 1),
  };
#else
  int2 offsets[6] = {
      int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 0), int2(1, 1), int2(0, 1),
  };
#endif

  f32 mip_level = log2(subcell_uv_step * f32(src_res));
  // Scalar loop
  [unroll] for (u32 tri_id = 0; tri_id < 2; tri_id++) {
    float2 suv0 = subcell_uv + offsets[0 + tri_id * 3] * subcell_uv_step;
    float2 suv1 = subcell_uv + offsets[1 + tri_id * 3] * subcell_uv_step;
    float2 suv2 = subcell_uv + offsets[2 + tri_id * 3] * subcell_uv_step;

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

    u32 num_samples = 0;

    // Bound the maximum triangle size to 32x32 pixels
    // imax.x          = imin.x + min(16, imax.x - imin.x);
    // imax.y          = imin.y + min(16, imax.y - imin.y);

    float3 normal_0 = normal_source.SampleLevel(ss, suv0, mip_level).xyz;
    float3 normal_1 = normal_source.SampleLevel(ss, suv1, mip_level).xyz;
    float3 normal_2 = normal_source.SampleLevel(ss, suv2, mip_level).xyz;

    // Still a scalar loop
    //[unroll(2)]
    for (i32 dy = 0; dy <= imax.y - imin.y; dy += 8) {
      //[unroll(2)]
      for (i32 dx = 0; dx <= imax.x - imin.x; dx += 8) {
        // Per lane iteration starts here
        // 64 lanes or 8x8 lanes
        u32 lane_x = WaveGetLaneIndex() & 0x7;
        u32 lane_y = WaveGetLaneIndex() >> 3;
        i32 x      = imin.x + dx + lane_x;
        i32 y      = imin.y + dy + lane_y;

        if (x >= 0 && y >= 0 && x < height && y < height) {
          float ef0 = init_ef0 + n0.x * float(dx + lane_x) + n0.y * float(dy + lane_y);
          float ef1 = init_ef1 + n1.x * float(dx + lane_x) + n1.y * float(dy + lane_y);
          float ef2 = init_ef2 + n2.x * float(dx + lane_x) + n2.y * float(dy + lane_y);

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
          // float2 pixel_uv = suv0 * b0 + suv1 * b1 + suv2 * b2;

          // add tid.x * 1.0e-6f to avoid z-fight
          u32 depth = u32((pp0.z) * 1000000);
          u32 next_depth;
          InterlockedMax(depth_target[int2(x, y)], depth, next_depth);
          if (depth > next_depth) {
            num_samples++;
            if (pc.flags & GI_RASTERIZATION_FLAG_PIXEL_COLOR_TRIANGLES)
              normal_target[int2(x, y)] =
                  float4(float2_splat(mip_level / 10.0f), random_color(suv0).z, 1.0f);
            else {
              float3 pixel_normal       = normalize(normal_0 * b0 + normal_1 * b1 + normal_2 * b2);
              normal_target[int2(x, y)] = float4(
                  float3_splat(max(0.0f, dot(pixel_normal, normalize(float3_splat(1.0f))))), 1.0f);
            }
          }
        }
      }
    }
    num_samples = WaveActiveSum(num_samples);
    // the first lane adds samples for the whole wavefront
    if (WaveGetLaneIndex() == 0) add_sample((suv0 + suv1 + suv2) / 3.0f, num_samples);
  }
} //

struct Vertex {
  float4 pos;
  float3 normal;
  float2 uv;
};

// Returns the number of visible pixels
// pp_i - position in clip space
u32 rasterize_triangle(RWTexture2D<float4> target, Vertex vtx0, Vertex vtx1, Vertex vtx2,
                       float mip_level) {
  // Target resolution
  uint width, height;
  target.GetDimensions(width, height);

  float4 pp0      = vtx0.pos;
  float4 pp1      = vtx1.pos;
  float4 pp2      = vtx2.pos;
  float3 normal_0 = vtx0.normal;
  float3 normal_1 = vtx1.normal;
  float3 normal_2 = vtx2.normal;

  // For simplicity just discard triangles that touch the boundary
  // @TODO(aschrein): Add proper clipping.
  if (!in_bounds(pp0) || !in_bounds(pp1) || !in_bounds(pp2)) return 0;

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
  float2 n0 = v1 - v0;
  float2 n1 = v2 - v1;
  float2 n2 = v0 - v2;

  // Double area
  float area2 = (n0.x * n2.y - n0.y * n2.x);

  // Back/small triangle culling
  if (area2 < 1.0e-6f) return 0;

  // 2D Edge Normals
  n0 = -float2(-n0.y, n0.x) / area2;
  n1 = -float2(-n1.y, n1.x) / area2;
  n2 = -float2(-n2.y, n2.x) / area2;

  // Bounding Box
  float2 fmin = float2(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)));
  float2 fmax = float2(max(v0.x, max(v1.x, v2.x)), max(v0.y, max(v1.y, v2.y)));

  int2 imin = int2(fmin);
  int2 imax = int2(fmax);

  imax.x = min(width - 1, max(0, imax.x));
  imax.y = min(height - 1, max(0, imax.y));

  // Edge function values at the first (imin.x + 0.5f, imin.y + 0.5f) sample position
  float2 first_sample = float2(imin) + float2(0.5f, 0.5f);
  float  init_ef0     = dot(first_sample - v0, n0);
  float  init_ef1     = dot(first_sample - v1, n1);
  float  init_ef2     = dot(first_sample - v2, n2);

  u32 num_samples = 0;

  // Bound the maximum triangle size to 8x8 pixels
  // imax.x = imin.x + min(8, imax.x - imin.x);
  // imax.y = imin.y + min(8, imax.y - imin.y);

  //[unroll(8)]
  //{
  //  i32 x = imin.x + (imax.x - imin.x) / 2;
  //  i32 y = imin.y + (imax.y - imin.y) / 2;
  //  // Barycentrics
  //  float b0 = 0.3f;
  //  float b1 = 0.3f;
  //  float b2 = 0.3f;
  //  // Perspective correction
  //  float bw    = b0 / pp0.w + b1 / pp1.w + b2 / pp2.w;
  //  b0          = b0 / pp0.w / bw;
  //  b1          = b1 / pp1.w / bw;
  //  b2          = b2 / pp2.w / bw;
  //  float depth = pp0.z * b0 + pp1.z * b1 + pp2.z * b2;
  //  // Per pixel Attributes
  //  // float2 pixel_uv = suv0 * b0 + suv1 * b1 + suv2 * b2;

  //  // add tid.x * 1.0e-6f to avoid z-fight
  //  u32 idepth = u32((depth)*1000000);
  //  u32 next_depth;
  //  InterlockedMax(depth_target[int2(x, y)], idepth, next_depth);
  //  if (idepth > next_depth) {
  //    num_samples++;
  //    if (pc.flags & GI_RASTERIZATION_FLAG_PIXEL_COLOR_TRIANGLES)
  //      // target[int2(x, y)] = float4(float2_splat(mip_level / 10.0f),
  //      // random_color(suv0).z, 1.0f);
  //      target[int2(x, y)] = float4(b0, b1, b2, 1.0f);
  //    else {
  //      float3 pixel_normal = normalize(normal_0 * b0 + normal_1 * b1 + normal_2 * b2);
  //      target[int2(x, y)] =
  //          float4(float3_splat(max(0.0f, dot(pixel_normal,
  //          normalize(float3_splat(1.0f))))), 1.0f);
  //    }
  //  }
  //}
#if 1
  for (i32 dy = 0; dy <= imax.y - imin.y; dy += 1) {
    float ef0 = init_ef0 + n0.y * float(dy);
    float ef1 = init_ef1 + n1.y * float(dy);
    float ef2 = init_ef2 + n2.y * float(dy);
    //[unroll(8)]
    for (i32 dx = 0; dx <= imax.x - imin.x; dx += 1) {
      i32 x = imin.x + dx;
      i32 y = imin.y + dy;
      if (ef0 > 0.0f && ef1 > 0.0f && ef2 > 0.0f) {
        // Barycentrics
        float b0 = ef1;
        float b1 = ef2;
        float b2 = ef0;
        // Perspective correction
        float bw    = b0 / pp0.w + b1 / pp1.w + b2 / pp2.w;
        b0          = b0 / pp0.w / bw;
        b1          = b1 / pp1.w / bw;
        b2          = b2 / pp2.w / bw;
        float depth = pp0.z * b0 + pp1.z * b1 + pp2.z * b2;
        // Per pixel Attributes
        float2 pixel_uv = vtx0.uv * b0 + vtx1.uv * b1 + vtx2.uv * b2;

        // add tid.x * 1.0e-6f to avoid z-fight
        u32 idepth = u32((depth)*1000000);
        u32 next_depth;
        InterlockedMax(depth_target[int2(x, y)], idepth, next_depth);
        if (idepth > next_depth) {
          num_samples++;
          if (pc.flags & GI_RASTERIZATION_FLAG_PIXEL_COLOR_TRIANGLES)
            // target[int2(x, y)] = float4(float2_splat(mip_level / 10.0f),
            // random_color(suv0).z, 1.0f);
            target[int2(x, y)] = float4(b0, b1, b2, 1.0f);
          else {
            // float3 pixel_normal = normalize(normal_0 * b0 + normal_1 * b1 + normal_2 * b2);
            float3 pixel_normal = normalize(normal_source.SampleLevel(ss, pixel_uv, mip_level).xyz);
            target[int2(x, y)]  = float4(
                float3_splat(max(0.0f, dot(pixel_normal, normalize(float3_splat(1.0f))))), 1.0f);
          }
        }
      }
      // Increment edge functions
      ef0 += n0.x;
      ef1 += n1.x;
      ef2 += n2.x;
    }
  }
#endif
  return num_samples;
}

// Returns the number of visible pixels
u32 gi_rasterize_quad(RWTexture2D<float4> target, Texture2D<float4> src_pos,
                      Texture2D<float4> src_normal, float2 uv0, float uv_size,
                      float4x4 obj_to_clip) {
  // Target resolution
  uint width, height;
  target.GetDimensions(width, height);

  // Resolution of the source images
  u32 src_res;
  {
    uint width, height;
    src_pos.GetDimensions(width, height);
    src_res = width;
  }
  f32 mip_level = log2(1.0f + uv_size * f32(src_res));

  int2 offsets[6] = {
      int2(0, 0), int2(0, 1), int2(1, 0), int2(1, 0), int2(0, 1), int2(1, 1),
  };
  u32 num_samples = 0;
  // Scalar loop
  [unroll] for (u32 tri_id = 0; tri_id < 2; tri_id++) {

    float2 suv0 = uv0 + offsets[0 + tri_id * 3] * uv_size;
    float2 suv1 = uv0 + offsets[1 + tri_id * 3] * uv_size;
    float2 suv2 = uv0 + offsets[2 + tri_id * 3] * uv_size;

    float4 pp0      = position_source.SampleLevel(ss, suv0, mip_level).xyzw;
    float4 pp1      = position_source.SampleLevel(ss, suv1, mip_level).xyzw;
    float4 pp2      = position_source.SampleLevel(ss, suv2, mip_level).xyzw;
    float3 normal_0 = src_normal.SampleLevel(ss, suv0, mip_level).xyz;
    float3 normal_1 = src_normal.SampleLevel(ss, suv1, mip_level).xyz;
    float3 normal_2 = src_normal.SampleLevel(ss, suv2, mip_level).xyz;
    pp0             = mul(obj_to_clip, float4(pp0.xyz, 1.0));
    pp1             = mul(obj_to_clip, float4(pp1.xyz, 1.0));
    pp2             = mul(obj_to_clip, float4(pp2.xyz, 1.0));
    Vertex vtx0;
    Vertex vtx1;
    Vertex vtx2;

    vtx0.pos    = pp0;
    vtx1.pos    = pp1;
    vtx2.pos    = pp2;
    vtx0.normal = normal_0;
    vtx1.normal = normal_1;
    vtx2.normal = normal_2;
    vtx0.uv     = suv0;
    vtx1.uv     = suv1;
    vtx2.uv     = suv2;
    num_samples += rasterize_triangle(target, vtx0, vtx1, vtx2, mip_level);
  }
  return num_samples;
}

[numthreads(GI_RASTERIZATION_GROUP_SIZE, GI_RASTERIZATION_GROUP_SIZE, 1)] //
    void
    main_tri_per_lane(uint3 tid
                      : SV_DispatchThreadID) {
      u32 cell_x = pc.cell_x;
      u32 cell_y = pc.cell_y;

      u32 cell_res;
      {
        IndirectArgs args = indirect_arg_buffer.Load<IndirectArgs>(
            12 * (pc.cell_x + pc.cell_y * COUNTER_GRID_RESOLUTION));
        cell_res = args.dimx;
      }
      if (cell_res == 0) return;
      float2 cell_uv = float2(cell_x, cell_y) / float(COUNTER_GRID_RESOLUTION);

      u32 subcell_x = tid.x;
      u32 subcell_y = tid.y;
      f32 subcell_uv_step =
          1.0f / float(cell_res * GI_RASTERIZATION_GROUP_SIZE * COUNTER_GRID_RESOLUTION);
      float2 subcell_uv = cell_uv + float2(subcell_x, subcell_y) * subcell_uv_step;
      u32 num_samples = gi_rasterize_quad(normal_target, position_source, normal_source, subcell_uv,
                                          subcell_uv_step, mul(fc.viewproj, pc.model));
      add_sample(subcell_uv + float2(subcell_uv_step, subcell_uv_step) / 2.0f, num_samples / 2);
    }