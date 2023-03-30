import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.lgamma(x)
print(out)
# [1.31452441, 1.76149750, 2.25271273, 1.09579802]