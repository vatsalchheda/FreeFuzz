import paddle

x = paddle.to_tensor([[1, -2j], [2j, 5]])
out_value = paddle.eigvalsh(x, UPLO='L')
print(out_value)
# Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [0.17157286, 5.82842731])