import paddle

x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
out = paddle.isfinite(x)
print(out)  # [False  True  True False  True False False]