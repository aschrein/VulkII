(scope
  (print "Scene 1 started")
  (add-mode rendering
    (print "Entering rendering mode")
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
    (let vs
      (create_shader
        (kind vertex)
        (uniform (type float3) (name vertex_offset))
        (input
          (location 0)
          (type float3)
          (name vertex_position))
        (output (type float2) (name tex_coords))
        (body
"""
@ENTRY
{
  float x = -1.0 + float((gl_VertexIndex & 1) << 2);
  float y = -1.0 + float((gl_VertexIndex & 2) << 1);
  tex_coords = float2(x * 0.5 + 0.5, y * 0.5 + 0.5);
  float4 pos = float4(vertex_position.xyz, 1);
  @EXPORT_POSITION<pos>
}
"""
        )
      )
    )
    (let ps
      (create_shader
        (kind pixel)
        (uniform (type float3) (name fragment_color))
        (input  (type float2) (name tex_coords))
        (output (type float4) (target 0))
        (body
"""
@ENTRY
{
  float4 f_color = float4(tex_coords, 0.0, 1.0) + float4(@UNIFORM<fragment_color>, 0.0);
  @EXPORT_COLOR0<f_color>
}
"""
        )
      )
    )
    (render-loop
      (start_frame)
      (start_render_pass "initial"
        (width  #)
        (height #)
        (add_render_target
          (name "initial/color")
          (rt_type COLOR)
          (format RGBA32_FLOAT)
        )
        (add_render_target
          (name "initial/depth")
          (rt_type DEPTH)
          (format D32_FLOAT)
        )
        (body
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
            (location 0)
            (binding  0)
            (offset   0)
            (format   RGB32_FLOAT)
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
          (clear_state)
        )
      )
      ; end rendering
;;      (print "next frame")
      (show_stats)
      (end_frame)
    )
    (release_resource vs)
    (release_resource ps)
    (release_resource vertex_buf)
    (release_resource index_buf)
  )
)
