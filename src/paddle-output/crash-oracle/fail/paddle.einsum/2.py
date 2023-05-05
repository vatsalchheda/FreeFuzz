import paddle
arg_1 = -34.0
arg_2_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.einsum(arg_1,arg_2,arg_3,)
