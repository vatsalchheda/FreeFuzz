import paddle
arg_class = paddle.nn.LeakyReLU()
arg_1_0_tensor = paddle.randint(0,2,[1, 32, 40851, 1])
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = arg_class(*arg_1)
