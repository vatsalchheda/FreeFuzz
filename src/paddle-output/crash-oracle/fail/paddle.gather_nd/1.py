import paddle
arg_1_tensor = paddle.randint(-1, 16384, [10, 21], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256, 128, [210, 2], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.gather_nd(arg_1,arg_2,)
