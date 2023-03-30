import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.reciprocal(x)
print(out)
# [-2.5        -5.         10.          3.33333333]