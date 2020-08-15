@(DECLARE_UNIFORM_BUFFER
  (set 1)
  (binding 0)
  (add_field (type float4x4)  (name view))
  (add_field (type float4x4)  (name viewproj))
  (add_field (type uint3)       (name density))
)
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 0)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_IMAGE
  (type READ_WRITE)
  (dim 2D)
  (set 0)
  (binding 1)
  (format R32_UINT)
  (name out_depth)
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 2)
  (format RGBA32_FLOAT)
  (name position_texture)
)
@(DECLARE_IMAGE
  (type SAMPLED)
  (dim 2D)
  (set 0)
  (binding 3)
  (format RGBA32_FLOAT)
  (name normal_texture)
)
@(DECLARE_SAMPLER
  (set 0)
  (binding 4)
  (name position_sampler)
)

@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  float2 uv = (float2(GLOBAL_THREAD_INDEX.xy) + float2_splat(0.5)) / float2(density.x, density.y);
  float2 duv = 
  float4 sp00 = texture(sampler2D(position_texture, position_sampler), uv);
  float4 sp01 = texture(sampler2D(position_texture, position_sampler), uv);
  float4 sp10 = texture(sampler2D(position_texture, position_sampler), uv);
  float4 sp00 = texture(sampler2D(position_texture, position_sampler), uv);
  float3 position = sp.xyz;
  if (sp.a < 1.0)
    return;
  float3 normal = texture(sampler2D(normal_texture, position_sampler), uv).xyz;

  float4 pp = mul4(viewproj, float4(position, 1.0));
  float4 nn = mul4(view, float4(normal, 0.0));
  //nn.xyz / nn.w;

  if (nn.z < 0.0)
    return;

  pp.xyz /= pp.w;
  
  if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
    return;
  i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
  i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
  if (pp.z > 0.0 && x > 0 && y > 0 && x < dim.x && y < dim.y) {
    float4 color = float4_splat(1.0)
      * (0.5 + 0.5 * dot(normalize(normal), normalize(float3(1.0, 1.0, 1.0))));
    u32 depth = u32(1.0 / pp.z);
    if (depth <= imageAtomicMin(out_depth, int2(x, y), depth)) {
      image_store(out_image, int2(x, y), color);
    }
  }
  
@(END)