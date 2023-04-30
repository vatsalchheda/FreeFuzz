import paddle
arg_1_tensor = paddle.randint(-256,8,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.sqrt(arg_1,)
