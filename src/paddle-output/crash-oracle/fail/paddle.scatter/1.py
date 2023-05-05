import paddle
arg_1_tensor = paddle.randint(-32, 1024, [1, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096, 16384, [60000], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32768, 4096, [60000], dtype=paddle.int64arg_3 = arg_3_tensor.clone()
res = paddle.scatter(arg_1,arg_2,arg_3,)
