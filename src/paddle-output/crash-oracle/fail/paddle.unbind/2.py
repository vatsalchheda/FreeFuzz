import paddle
arg_1_tensor = paddle.randint(-8192,16384,[16, 0, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.unbind(arg_1,axis=arg_2,)
