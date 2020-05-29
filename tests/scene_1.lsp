(scope
  (print "Scene 1 started")
  (add-mode rendering
    (print "Entering rendering mode")
    (render-loop
      (start_frame)
      (print "next frame")
      (end_frame)
    )
  )
)
