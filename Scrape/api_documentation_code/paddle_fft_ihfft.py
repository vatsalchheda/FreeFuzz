import paddle

spectrum = paddle.to_tensor([10.0, -5.0, 0.0, -1.0, 0.0, -5.0])
print(paddle.fft.ifft(spectrum))
# Tensor(shape=[6], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#       [(-0.1666666716337204+0j),  (1-1.9868215517249155e-08j), (2.3333334922790527-1.9868215517249155e-08j),  (3.5+0j), (2.3333334922790527+1.9868215517249155e-08j),  (1+1.9868215517249155e-08j)])
print(paddle.fft.ihfft(spectrum))
#  Tensor(shape = [4], dtype = complex64, place = CUDAPlace(0), stop_gradient = True,
#         [(-0.1666666716337204+0j),  (1-1.9868215517249155e-08j), (2.3333334922790527-1.9868215517249155e-08j),  (3.5+0j)])