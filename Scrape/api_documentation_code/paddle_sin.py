import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.sin(x)
print(out)
# [-0.38941834 -0.19866933  0.09983342  0.29552021]