import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.exp(x)
print(out)
# [0.67032005 0.81873075 1.10517092 1.34985881]