import paddle

# scale input
beta = paddle.distribution.Beta(alpha=0.5, beta=0.5)
print(beta.mean)
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [0.50000000])
print(beta.variance)
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [0.12500000])
print(beta.entropy())
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [0.12500000])

# tensor input with broadcast
beta = paddle.distribution.Beta(alpha=paddle.to_tensor([0.2, 0.4]), beta=0.6)
print(beta.mean)
# Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [0.25000000, 0.40000001])
print(beta.variance)
# Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [0.10416666, 0.12000000])
print(beta.entropy())
# Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [-1.91923141, -0.38095069])