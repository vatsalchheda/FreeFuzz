import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.atanh(x)
print(out)
# [-0.42364895, -0.20273256,  0.10033535,  0.30951962]