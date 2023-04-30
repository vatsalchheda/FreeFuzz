import paddle
arg_1_tensor = paddle.randint(-16384,2048,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,4,[], dtype=paddle.complex64)
arg_2 = arg_2_tensor.clone()
res = paddle.assign(arg_1,arg_2,)
