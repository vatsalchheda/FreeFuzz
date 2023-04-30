import paddle
arg_1_tensor = paddle.randint(-16384,8192,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.sigmoid(arg_1,)
