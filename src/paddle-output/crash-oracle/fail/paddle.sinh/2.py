import paddle
arg_1_tensor = paddle.randint(-2048,64,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.sinh(arg_1,)
