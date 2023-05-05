import paddle
arg_1_tensor = paddle.rand([6, 4, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 47.0
arg_3 = -36.0
res = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
