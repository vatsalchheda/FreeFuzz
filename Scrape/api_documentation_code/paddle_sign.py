import paddle

x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
out = paddle.sign(x=x)
print(out)  # [1.0, 0.0, -1.0, 1.0]