import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.asinh(x)
print(out)
# [-0.39003533, -0.19869010,  0.09983408,  0.29567307]