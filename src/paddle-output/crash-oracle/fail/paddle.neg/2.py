import paddle
arg_1_tensor = paddle.randint(-2048,512,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.neg(arg_1,)
