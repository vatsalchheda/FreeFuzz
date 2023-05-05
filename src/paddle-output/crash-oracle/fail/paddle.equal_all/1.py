import paddle
arg_1_tensor = paddle.randint(-64, 64, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4, 128, [3], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.equal_all(arg_1,arg_2,)
