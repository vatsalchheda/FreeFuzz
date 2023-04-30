import paddle
arg_1_tensor = paddle.randint(-16384,32,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.log(arg_1,)
