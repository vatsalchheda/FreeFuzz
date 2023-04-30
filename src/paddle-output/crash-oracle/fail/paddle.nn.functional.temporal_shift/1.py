import paddle
arg_1_tensor = paddle.randint(-2048,512,[6, 4, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -56.8
res = paddle.nn.functional.temporal_shift(x=arg_1,seg_num=arg_2,shift_ratio=arg_3,)
