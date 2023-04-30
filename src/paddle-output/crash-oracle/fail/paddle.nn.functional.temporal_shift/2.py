import paddle
arg_1_tensor = paddle.randint(-16,8192,[6, 4, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = 0.2
res = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
