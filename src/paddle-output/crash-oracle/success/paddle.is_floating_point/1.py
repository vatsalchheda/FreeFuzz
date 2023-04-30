import paddle
arg_1_tensor = paddle.randint(-128,32768,[4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.is_floating_point(arg_1,)
