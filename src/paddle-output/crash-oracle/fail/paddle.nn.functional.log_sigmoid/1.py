import paddle
arg_1_tensor = paddle.randint(-1024,8192,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.log_sigmoid(arg_1,)
