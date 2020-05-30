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
      (print "next frame")
      (show_stats)
      (end_frame)
    )
  )
)
