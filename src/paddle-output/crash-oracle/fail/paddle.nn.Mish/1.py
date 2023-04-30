import paddle
arg_class = paddle.nn.Mish()
arg_1_0_tensor = paddle.randint(-64,4096,[3], dtype=paddle.int16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = arg_class(*arg_1)
