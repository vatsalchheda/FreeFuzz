import paddle

x = paddle.to_tensor([(2+2j), (2+2j), (3+3j)])
hfftn_x = paddle.fft.hfftn(x)
print(hfftn_x)
# Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [ 9.,  3.,  1., -5.])