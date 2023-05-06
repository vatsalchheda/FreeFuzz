import paddle
arg_1_tensor = paddle.randint(-1, 1024, [4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32, 1024, [4], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.floor_divide(arg_1,arg_2,)
