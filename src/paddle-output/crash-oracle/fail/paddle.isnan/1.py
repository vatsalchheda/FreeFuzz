import paddle
arg_1_tensor = paddle.rand([-1, 256, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.isnan(arg_1,)
