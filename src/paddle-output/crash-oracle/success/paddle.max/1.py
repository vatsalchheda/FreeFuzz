import paddle
arg_1_tensor = paddle.randint(-2, 8192, [10], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.max(arg_1,)
