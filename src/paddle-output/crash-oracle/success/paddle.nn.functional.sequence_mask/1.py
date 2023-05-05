import paddle
arg_1_tensor = paddle.randint(-8, 2, [4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 12
arg_3 = "int64"
arg_4 = None
res = paddle.nn.functional.sequence_mask(arg_1,arg_2,arg_3,arg_4,)
