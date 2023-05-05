import paddle
arg_1_tensor = paddle.rand([1, 30001], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 32768, [4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.index_sample(arg_1,arg_2,)
