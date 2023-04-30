import paddle
arg_1_tensor = paddle.randint(0,2,[2, 2], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = None
res = paddle.repeat_interleave(arg_1,arg_2,arg_3,)
