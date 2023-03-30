import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.tan(x)
print(out)
# [-0.42279324, -0.20271005, 0.10033467, 0.30933627]