import paddle
arg_1_tensor = paddle.randint(-32,16,[63, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 14
res = paddle.tril(arg_1,diagonal=arg_2,)
