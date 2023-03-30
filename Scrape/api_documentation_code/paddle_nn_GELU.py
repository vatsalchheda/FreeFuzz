import paddle

x = paddle.to_tensor([[-1, 0.5],[1, 1.5]])

m = paddle.nn.GELU()
out = m(x) # [-0.158655 0.345731 0.841345 1.39979]

m = paddle.nn.GELU(True)
out = m(x) # [-0.158808 0.345714 0.841192 1.39957]