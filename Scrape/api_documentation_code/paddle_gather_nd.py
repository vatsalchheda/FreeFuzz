import paddle

x = paddle.to_tensor([[[1, 2], [3, 4], [5, 6]],
                      [[7, 8], [9, 10], [11, 12]]])
index = paddle.to_tensor([[0, 1]])

output = paddle.gather_nd(x, index) #[[3, 4]]