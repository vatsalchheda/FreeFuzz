import paddle
arg_1_tensor = paddle.randint(-8, 1, [4], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
res = paddle.sum(arg_1,)
