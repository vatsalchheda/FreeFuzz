import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768, 8192, [2], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
res = paddle.linalg.lu_unpack(arg_1,arg_2,)
