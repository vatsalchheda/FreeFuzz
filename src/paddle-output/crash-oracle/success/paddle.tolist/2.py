import paddle
arg_1_tensor = paddle.randint(-32768,1,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.tolist(arg_1,)
