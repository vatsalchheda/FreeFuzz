import paddle
arg_1_tensor = paddle.randint(-1024,4096,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.increment(arg_1,)
