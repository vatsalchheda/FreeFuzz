import paddle
arg_1_tensor = paddle.randint(-2048, 2048, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2, 4, [3], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.bitwise_xor(arg_1,arg_2,)
