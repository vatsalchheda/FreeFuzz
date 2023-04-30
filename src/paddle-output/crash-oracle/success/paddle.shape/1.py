import paddle
arg_1_tensor = paddle.randint(-32,64,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.shape(arg_1,)
