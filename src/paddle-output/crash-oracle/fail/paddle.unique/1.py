import paddle
arg_1_tensor = paddle.randint(-2,32768,[7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.unique(arg_1,)
