import paddle
arg_1_tensor = paddle.randint(-2, 128, [0], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -32
res = paddle.repeat_interleave(arg_1,arg_2,arg_3,)
