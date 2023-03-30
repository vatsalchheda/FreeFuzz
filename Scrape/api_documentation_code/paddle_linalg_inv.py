import paddle

mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
inv = paddle.inverse(mat)
print(inv) # [[0.5, 0], [0, 0.5]]