import paddle
arg_1_tensor = paddle.randint(-4,16,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.erf(arg_1,)
