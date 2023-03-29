import paddle

data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
res = paddle.digamma(data)
print(res)
# Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [[-0.57721591,  0.03648996],
#        [ nan       ,  5.32286835]])