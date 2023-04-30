import paddle
arg_1_tensor = paddle.randint(-8192,4,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.tan(arg_1,)
