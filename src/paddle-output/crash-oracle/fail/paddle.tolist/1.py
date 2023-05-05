import paddle
arg_1_tensor = paddle.randint(-256, 2048, [52], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.tolist(arg_1,)
