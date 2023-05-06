import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128, 32768, [2], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
res = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)
