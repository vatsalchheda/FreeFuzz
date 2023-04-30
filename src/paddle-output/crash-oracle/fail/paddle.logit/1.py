import paddle
arg_1_tensor = paddle.randint(-16384,128,[3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.logit(arg_1,)
