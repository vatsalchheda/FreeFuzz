import paddle
arg_1_tensor = paddle.randint(-512,512,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.zero_(arg_1,)
