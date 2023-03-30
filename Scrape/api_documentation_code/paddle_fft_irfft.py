import paddle

x = paddle.to_tensor([1, -1j, -1])
irfft_x = paddle.fft.irfft(x)
print(irfft_x)
# Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [0., 1., 0., 0.])