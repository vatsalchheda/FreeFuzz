import paddle
arg_1_tensor = paddle.rand([1, 64, 314], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.floor(arg_1,)
