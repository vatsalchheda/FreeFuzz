import paddle

input = paddle.rand([2,2],'float32')
print(input)
# Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [[0.02331470, 0.42374918],
#         [0.79647720, 0.74970269]])

output = paddle.trunc(input)
print(output)
# Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#         [[0., 0.],
#         [0., 0.]]))