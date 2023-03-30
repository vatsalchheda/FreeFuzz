import paddle

x = paddle.arange(48, dtype="float32").reshape((2, 6, 2, 2))
group_norm = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
group_norm_out = group_norm(x)

print(group_norm_out)