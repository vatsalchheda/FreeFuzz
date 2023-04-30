import paddle
arg_1_tensor = paddle.randint(-8192,1024,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.trace(arg_1,)
