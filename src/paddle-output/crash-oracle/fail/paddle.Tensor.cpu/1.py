import paddle
arg_1_tensor = paddle.randint(-8192,4096,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.cpu(arg_1,)
