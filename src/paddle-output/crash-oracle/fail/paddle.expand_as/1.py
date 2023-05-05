import paddle
arg_1_tensor = paddle.randint(-256, 64, [3], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.expand_as(arg_1,arg_2,)
