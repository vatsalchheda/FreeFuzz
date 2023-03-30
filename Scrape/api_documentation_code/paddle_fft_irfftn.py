import paddle

x = paddle.to_tensor([2.+2.j, 2.+2.j, 3.+3.j]).astype(paddle.complex128)
print(x)
irfftn_x = paddle.fft.irfftn(x)
print(irfftn_x)

# Tensor(shape=[3], dtype=complex128, place=Place(cpu), stop_gradient=True,
#        [(2+2j), (2+2j), (3+3j)])
# Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [ 2.25000000, -1.25000000,  0.25000000,  0.75000000])