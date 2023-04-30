import paddle
arg_1_tensor = paddle.randint(-32768,8192,[8], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.unique_consecutive(arg_1,)
