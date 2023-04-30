import paddle
arg_1_tensor = paddle.randint(-2,8,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,1,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.tile(arg_1,repeat_times=arg_2,)
