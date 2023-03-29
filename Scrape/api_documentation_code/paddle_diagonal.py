import paddle

x = paddle.rand([2,2,3],'float32')
print(x)
# Tensor(shape=[2, 2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[[0.45661032, 0.03751532, 0.90191704],
#          [0.43760979, 0.86177313, 0.65221709]],

#         [[0.17020577, 0.00259554, 0.28954273],
#          [0.51795638, 0.27325270, 0.18117726]]])

out1 = paddle.diagonal(x)
print(out1)
#Tensor(shape=[3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [[0.45661032, 0.51795638],
#        [0.03751532, 0.27325270],
#        [0.90191704, 0.18117726]])

out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
print(out2)
#Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [[0.45661032, 0.86177313],
#        [0.17020577, 0.27325270]])

out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
print(out3)
#Tensor(shape=[3, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [[0.43760979],
#        [0.86177313],
#        [0.65221709]])

out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
print(out4)
#Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#       [[0.45661032, 0.86177313],
#        [0.17020577, 0.27325270]])