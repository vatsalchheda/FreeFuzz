import paddle

x = paddle.to_tensor([0.0, 1.0, 0.0, 0.0])
print(paddle.fft.rfft(x))
# Tensor(shape=[3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
#        [ (1+0j), -1j    , (-1+0j)])