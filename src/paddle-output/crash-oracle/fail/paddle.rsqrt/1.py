import paddle
arg_1_tensor = paddle.randint(-128,64,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.rsqrt(arg_1,)
