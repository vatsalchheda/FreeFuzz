import paddle

x = paddle.to_tensor([0, 0.5, -1.], dtype="float32")
out = paddle.erfinv(x)
# out: [0, 0.4769, -inf]