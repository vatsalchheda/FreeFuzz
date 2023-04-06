import paddle

x1 = paddle.to_tensor(12)
x2 = paddle.to_tensor(20)
paddle.gcd(x1, x2)
# Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [4])

x3 = paddle.arange(6)
paddle.gcd(x3, x2)
# Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [20, 1 , 2 , 1 , 4 , 5])

x4 = paddle.to_tensor(0)
paddle.gcd(x4, x2)
# Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [20])

paddle.gcd(x4, x4)
# Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [0])

x5 = paddle.to_tensor(-20)
paddle.gcd(x1, x5)
# Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [4])