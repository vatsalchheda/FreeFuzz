import paddle
arg_1_0_tensor = paddle.randint(-1, 4096, [1], dtype=paddle.int32arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = paddle.stack(arg_1,)
