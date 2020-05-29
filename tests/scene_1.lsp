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
            Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER)
          (flags
            Memory_Bits::MAPPABLE)
          512
        )
      )
      (release_resource tmp_buf)
      (print "next frame")
      (show_stats)
      (end_frame)
    )
  )
)
