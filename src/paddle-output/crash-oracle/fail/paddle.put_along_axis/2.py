import paddle
arg_1_tensor = paddle.randint(-16, 4096, [2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1, 8192, [1, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 111
arg_4 = -24
res = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
