import paddle
arg_1_tensor = paddle.randint(-32,16,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.cumprod(arg_1,dim=arg_2,)
