import paddle
arg_1_tensor = paddle.randint(-1024,512,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.reciprocal(arg_1,)
