import paddle
arg_1_tensor = paddle.randint(-16384, 1, [100], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048, 32, [200], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.meshgrid(arg_1,arg_2,)
