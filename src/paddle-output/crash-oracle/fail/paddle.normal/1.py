import paddle
arg_1_tensor = paddle.randint(-1024,4096,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.normal(mean=arg_1,)
