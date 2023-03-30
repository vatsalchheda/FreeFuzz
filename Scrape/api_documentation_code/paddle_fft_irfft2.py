import paddle

x = paddle.to_tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
irfft2_x = paddle.fft.irfft2(x)
print(irfft2_x)
# Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[ 2.37500000, -1.12500000,  0.37500000,  0.87500000],
#         [ 0.12500000,  0.12500000,  0.12500000,  0.12500000]])