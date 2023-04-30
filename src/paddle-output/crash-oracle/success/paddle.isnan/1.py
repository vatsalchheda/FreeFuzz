import paddle
arg_1_tensor = paddle.randint(-128,1024,[12], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.isnan(arg_1,)
