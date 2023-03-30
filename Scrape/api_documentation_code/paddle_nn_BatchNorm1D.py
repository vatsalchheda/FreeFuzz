import paddle

x = paddle.rand((2, 1, 3))
batch_norm = paddle.nn.BatchNorm1D(1)
batch_norm_out = batch_norm(x)

print(batch_norm_out)