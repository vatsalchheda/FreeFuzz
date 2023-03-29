import paddle

x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
out1 = paddle.clip(x1, min=3.5, max=5.0)
out2 = paddle.clip(x1, min=2.5)
print(out1)
# [[3.5, 3.5]
# [4.5, 5.0]]
print(out2)
# [[2.5, 3.5]
# [[4.5, 6.4]