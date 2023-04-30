import paddle
arg_1_tensor = paddle.randint(-4096,16384,[3, 4, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.unbind(arg_1,axis=arg_2,)
