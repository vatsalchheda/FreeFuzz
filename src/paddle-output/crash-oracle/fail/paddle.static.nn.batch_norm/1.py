import paddle
arg_1_tensor = paddle.randint(-8, 8192, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.static.nn.batch_norm(arg_1,)
