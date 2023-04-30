import paddle
arg_1_tensor = paddle.randint(-32,2,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3 = None
res = paddle.repeat_interleave(arg_1,arg_2,arg_3,)
