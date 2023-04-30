import paddle
arg_1_tensor = paddle.randint(-16,256,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.value(arg_1,)
