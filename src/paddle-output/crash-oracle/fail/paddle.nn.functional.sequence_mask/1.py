import paddle
arg_1_tensor = paddle.randint(-8, 64, [4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 12
arg_3 = "circular"
arg_4 = None
res = paddle.nn.functional.sequence_mask(arg_1,arg_2,arg_3,arg_4,)
