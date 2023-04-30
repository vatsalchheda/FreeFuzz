import paddle
arg_1_tensor = paddle.randint(-4096,8,[6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.floor(arg_1,)
