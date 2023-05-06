import paddle
arg_1_tensor = paddle.randint(-32768, 128, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 8, [3], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.not_equal(arg_1,arg_2,)
