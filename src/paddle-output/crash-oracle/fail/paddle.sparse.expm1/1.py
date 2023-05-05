import paddle
arg_1_tensor = paddle.rand([3, 43], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.sparse.expm1(arg_1,)
