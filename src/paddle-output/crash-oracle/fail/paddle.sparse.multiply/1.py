import paddle
arg_1_tensor = paddle.randint(-32768, 128, [4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.sparse.multiply(arg_1,arg_2,)
