import paddle
arg_1_0_tensor = paddle.randint(-8192,16384,[32, 1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2 = None
res = paddle.nn.functional.mish(arg_1,arg_2,)
