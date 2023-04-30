import paddle
arg_1_tensor = paddle.randint(-2,1,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.asin(arg_1,)
