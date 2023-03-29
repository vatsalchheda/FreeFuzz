  # Example 1 (0-D tensor)
  x = paddle.to_tensor([0.79])
  paddle.t(x) # [0.79]

  # Example 2 (1-D tensor)
  x = paddle.to_tensor([0.79, 0.84, 0.32])
  paddle.t(x) # [0.79000002, 0.83999997, 0.31999999]
  paddle.t(x).shape # [3]

  # Example 3 (2-D tensor)
  x = paddle.to_tensor([[0.79, 0.84, 0.32],
                       [0.64, 0.14, 0.57]])
  x.shape # [2, 3]
  paddle.t(x)
  # [[0.79000002, 0.63999999],
  #  [0.83999997, 0.14000000],
  #  [0.31999999, 0.56999999]]
  paddle.t(x).shape # [3, 2]