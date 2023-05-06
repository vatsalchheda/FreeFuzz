import paddle
arg_1 = 2090
arg_class = paddle.nn.BatchNorm2D(arg_1,)
arg_2_0_tensor = paddle.randint(-16384, 512, [1, 153, 271, 32], dtype=paddle.int64arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
