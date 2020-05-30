(scope
  (print "Scene 1 started")
  (add-mode rendering
    (print "Entering rendering mode")
    (render-loop
      (start_frame)
      (let vertex_buf
        (create_buffer
          (flags
            Buffer_Usage_Bits::USAGE_TRANSIENT
            Buffer_Usage_Bits::USAGE_VERTEX_BUFFER)
          (flags
            Memory_Bits::MAPPABLE)
          512
        )
      )
      (let *buf (map_buffer vertex_buf))
      (memcpy *buf
        (array_f32
          -1.0 -1.0 0.0
           0.0  1.0 0.0
           1.0  1.0 0.0
        ))
      (unmap_buffer vertex_buf)
      (let index_buf
        (create_buffer
          (flags
            Buffer_Usage_Bits::USAGE_TRANSIENT
            Buffer_Usage_Bits::USAGE_INDEX_BUFFER)
          (flags
            Memory_Bits::MAPPABLE)
          512
        )
      )
      (let *buf (map_buffer index_buf))
      (memcpy *buf
        (array_i32
          0 1 2
        ))
      (unmap_buffer index_buf)
      (let tmp_cs_text
        (create_shader
          (header
"""
#extension GL_KHR_shader_subgroup_vote    : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
"""
          )
          (kind compute)
          (size 8 8 8)

          (uniform
            (format R16_FLOAT)
            (type image2D)
            (name out_image)
            (count 100)
            (freq PER_DC))

          (uniform (type texture2D) (name in_image)       (freq PER_DC))
          (uniform (type sampler)   (name in_sampler)     (freq PER_DC))
          (uniform (type float3)    (name lpv_min)        (freq PER_DC))
          (uniform (type float3)    (name lpv_max)        (freq PER_DC))
          (uniform (type float3)    (name lpv_cell_size)  (freq PER_DC))
          (uniform (type uint3)     (name lpv_size)       (freq PER_DC))
          (uniform (type float4x4)  (name rsm_viewproj)   (freq PER_DC) (layout row_major))
          (uniform (type float3)    (name rsm_pos)        (freq PER_DC))
          (uniform (type float3)    (name rsm_y)          (freq PER_DC))
          (uniform (type float3)    (name rsm_x)          (freq PER_DC))
          (uniform (type float3)    (name rsm_z)          (freq PER_DC))
          (uniform (type uint)      (name level)          (freq PER_DC))
          (uniform (type uint)      (name max_level)      (freq PER_DC))
          (uniform (type uint)      (name mode)           (freq PER_DC))
          (body
"""
// Copy image to the top mip level
const int MODE_COPY = 0;
// Blur the prev mip level to the next
const int MODE_BLUR = 1;


// Src: http://roar11.com/2015/07/screen-space-glossy-reflections/
const int2 h_offsets[7] = {{-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}};
const int2 v_offsets[7] = {{0, -3}, {0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {0, 3}};
const float weights[7] = {0.001f, 0.028f, 0.233f, 0.474f, 0.233f, 0.028f, 0.001f};

float4 load_premult_alpha(int tex_id, int2 coord) {
  int2 dim = imageSize(@UNIFORM<out_image>[tex_id]);
  if (coord.x < 0)
    coord.x = 0;
  if (coord.x >= dim.x)
    coord.x = dim.x - 1;
  if (coord.y < 0)
    coord.y = 0;
  if (coord.y >= dim.y)
    coord.y = dim.y - 1;
  float4 c = imageLoad(@UNIFORM<out_image>[tex_id], coord);
  return float4(c.xyz * c.w, c.w);
}

@ENTRY
{
    int2 dim = imageSize(@UNIFORM<out_image>[@UNIFORM<level>]);
    float2 uv = float2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    if (@UNIFORM<mode> == MODE_COPY) {
        imageStore(@UNIFORM<out_image>[@UNIFORM<level>], int2(gl_GlobalInvocationID.xy),
          texelFetch(in_image, int2(gl_GlobalInvocationID.xy), 0)
        );
    } else if (@UNIFORM<mode> == MODE_BLUR) {
        int2 pxy = int2(gl_GlobalInvocationID.xy) * 2;
        // @TODO: it's not a real blur just a first iteration to have something
        float4 final_val = float4(0.0);
        for (int i = 0; i < 7; i++) {
          final_val += load_premult_alpha(int(@UNIFORM<level>) - 1, pxy + h_offsets[i])
                        * weights[i];
        }
        for (int i = 0; i < 7; i++) {
          final_val += load_premult_alpha(int(@UNIFORM<level>) - 1, pxy + v_offsets[i] + int2(1, 1))
                        * weights[i];
        }
        final_val *= 0.5;
        imageStore(@UNIFORM<out_image>[@UNIFORM<level>], int2(gl_GlobalInvocationID.xy),
          final_val
        );
    }
}
"""
          )
        )
      )
      (let vs
        (create_shader
          (kind vertex)
          (input  (type float3) (name vertex_position))
          (output (type float2) (name tex_coords))
          (body
"""
@ENTRY
{
  float x = -1.0 + float((gl_VertexIndex & 1) << 2);
  float y = -1.0 + float((gl_VertexIndex & 2) << 1);
  tex_coords = float2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  float4 pos = float4(vertex_position.xyz, 1);
  @EXPORT_POSITION(pos)
}
"""
          )
        )
      )
      (let ps
        (create_shader
          (kind pixel)
          (input  (type float2) (name tex_coords))
          (output (type float4) (name f_color) (target 0))
          (body
"""
@ENTRY
{
  f_color = float4(tex_coords, 0.0, 1.0);
  @EXPORT(f_color)
}
"""
          )
        )
      )
      ; start rendering
      (VS_bind_shader vs)
      (PS_bind_shader ps)
      (IA_set_topology TRIANGLE_LIST)
      (IA_bind_index_buffer index_buf
        (offset 0)
        (format R32_UINT)
      )
      (IA_bind_vertex_buffer vertex_buf
        (binding 0)
        (offset  0)
        (stride  12)
        (rate    PER_VERTEX)
      )
      (IA_bind_attribute
        (location
          (shader_get_input_location vs "vertex_position"))
        (binding 0)
        (offset  0)
        (format  RGB32_FLOAT)
      )
      (IA_set_cull_mode
        (front_face CW)
        (cull_mode NONE)
      )
      (draw_indexed_instanced
        (index_count    3)
        (instance_count 1)
        (start_index    0)
        (start_vertex   0)
        (start_instace  0)
      )
      ; end rendering
      (release_resource vs)
      (release_resource ps)
      (release_resource vertex_buf)
      (release_resource index_buf)
;;      (print "next frame")
      (show_stats)
      (end_frame)
    )
  )
)
