import paddle

x = paddle.to_tensor([1, 2, 3], dtype='float32')

# example 1: y is a float or int
res = paddle.pow(x, 2)
print(res)
# Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1., 4., 9.])
res = paddle.pow(x, 2.5)
print(res)
# Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1.         , 5.65685415 , 15.58845711])

# example 2: y is a Tensor
y = paddle.to_tensor([2], dtype='float32')
res = paddle.pow(x, y)
print(res)
# Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1., 4., 9.])