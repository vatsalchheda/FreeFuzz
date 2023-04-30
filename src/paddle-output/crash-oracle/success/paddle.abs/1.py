import paddle
arg_1_tensor = paddle.randint(-64,1024,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.abs(arg_1,)
