import paddle

x = paddle.to_tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
hfft2_x = paddle.fft.hfft2(x)
print(hfft2_x)
# Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [[19.,  7.,  3., -9.],
#         [ 1.,  1.,  1.,  1.]])