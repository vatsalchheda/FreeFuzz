import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.cos(x)
print(out)
# [0.92106099 0.98006658 0.99500417 0.95533649]