import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.tanh(x)
print(out)
# [-0.37994896 -0.19737532  0.09966799  0.29131261]