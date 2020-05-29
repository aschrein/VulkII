(scope
  (print (format "hello world %i" 1))
  (add-mode rendering
    (print (format "hello world %i" 2))
    (render-loop
      (start_frame)
      (end_frame)
    )
  )
)
