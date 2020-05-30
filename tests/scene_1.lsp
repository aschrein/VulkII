(scope
  (print "Scene 1 started")
  (add-mode rendering
    (print "Entering rendering mode")
    (render-loop
      (start_frame)
      (let tmp_buf
        (create_buffer
          (flags
            Buffer_Usage_Bits::USAGE_TRANSIENT
            Buffer_Usage_Bits::USAGE_VERTEX_BUFFER)
          (flags
            Memory_Bits::MAPPABLE)
          512
        )
      )
      (let *buf (map_buffer tmp_buf))
      (memcpy *buf
        (array_f32
          -1.0 -1.0 0.0
           0.0  1.0 0.0
           1.0  1.0 0.0
        ))
      (unmap_buffer tmp_buf)
      (release_resource tmp_buf)
      (let cs_text
        (build_shader
          (type pixel)
          (input vec3 in_position)
          (input vec3 in_normal)
          (input vec3 in_tangent)
          (input vec3 in_binormal)
          (input vec2 in_texcoord)

          (output vec4 g_albedo)
          (output vec4 g_normal)
          (output vec4 g_arm)

          (push_constants push_constants
            (member mat4 transform)
            (member int albedo_id)
            (member int normal_id)
            (member int arm_id)
            (member float metal_factor)
            (member float roughness_factor)
            (member vec4 albedo_factor)
          )
          (set
            (uniform_buffer uniforms
              (member mat4 view)
              (member mat4 proj)
              (member vec3 camera_pos)
            )
          )
          (set
            (uniform_array
              sampler2D
              textures
              4096
            )
          )
          (body
"""
void main() {
  vec4 out_albedo;
  vec4 out_normal;
  vec4 out_arm;
  if (push_constants.albedo_id >= 0) {
    vec4 s0 = texture(textures[nonuniformEXT(push_constants.albedo_id)], in_texcoord, -1.0);
//    s0 = pow(s0, vec4(2.2));
    out_albedo = push_constants.albedo_factor * s0;
  } else {
    out_albedo = push_constants.albedo_factor;
  }
  if (out_albedo.a < 0.5)
    discard;
  vec3 normal = normalize(in_normal).xyz;
  if (push_constants.normal_id >= 0) {
    vec3 tangent = normalize(in_tangent).xyz;
    vec3 binormal = normalize(in_binormal).xyz;
    vec3 nc = texture(textures[nonuniformEXT(push_constants.normal_id)], in_texcoord, -1.0).xyz;
    out_normal = vec4(
    normalize(
      normal * nc.z +
    tangent * (2.0 * nc.x - 1.0) +
    binormal * (2.0 * nc.y - 1.0)), 0.0);
  } else {
    out_normal = vec4(normal, 0.0);
  }
  if (push_constants.arm_id >= 0) {
    vec4 s0 = texture(textures[nonuniformEXT(push_constants.arm_id)], in_texcoord, -2.0);
//    s0 = pow(s0, vec4(2.2));
    out_arm = vec4(1.0, push_constants.roughness_factor, push_constants.metal_factor, 1.0f) * s0;
  } else {
    out_arm = vec4(1.0, push_constants.roughness_factor, push_constants.metal_factor, 1.0f);
  }
  g_albedo = out_albedo;
  g_normal = vec4(out_normal.xyz, length(in_position - uniforms.camera_pos));
  g_arm = out_arm;
}
"""
          )
        )
      )
      (let cs (compile_shader cs_text))
      (release_resource cs)
      (print "next frame")
      (show_stats)
      (end_frame)
    )
  )
)
