import paddle
arg_1_tensor = paddle.randint(-32,32768,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.as_real(arg_1,)
