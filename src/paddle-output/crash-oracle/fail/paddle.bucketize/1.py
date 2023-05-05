import paddle
arg_1_tensor = paddle.randint(-64, 4096, [2, 4], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([58, 1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.bucketize(arg_1,arg_2,)
