import paddle

x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')
#Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [-1,  1,  1, -1])

y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')
#Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [-1,  -1,  1, 1])

out = paddle.atan2(x, y)
#Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [-2.35619450,  2.35619450,  0.78539819, -0.78539819])