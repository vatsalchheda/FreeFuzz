import paddle
arg_1_tensor = paddle.randint(-4, 4096, [26], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.sum(arg_1,)
