import paddle
arg_1_tensor = paddle.rand([10], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,256,[2, 1, 4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.equal_all(arg_1,arg_2,)
