import paddle
arg_1_tensor = paddle.randint(-2, 32768, [1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.empty(arg_1,)
