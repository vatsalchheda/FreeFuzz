import paddle

dirichlet = paddle.distribution.Dirichlet(paddle.to_tensor([1., 2., 3.]))

print(dirichlet.entropy())
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [-1.24434423])
print(dirichlet.prob(paddle.to_tensor([.3, .5, .6])))
# Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [10.80000114])