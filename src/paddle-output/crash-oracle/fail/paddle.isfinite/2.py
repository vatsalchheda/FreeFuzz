import paddle
arg_1_tensor = paddle.randint(-32768,32768,[1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.isfinite(arg_1,)
