import paddle
arg_1_tensor = paddle.randint(-2048, 512, [4], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
res = paddle.is_floating_point(arg_1,)
