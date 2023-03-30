import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
m = paddle.nn.LogSigmoid()
out = m(x) # [-0.313262 -0.126928 -0.0485874 -0.0181499]