import paddle

x = paddle.to_tensor([1, -1j, -1])
hfft_x = paddle.fft.hfft(x)
print(hfft_x)
# Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [0., 0., 0., 4.])