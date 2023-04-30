import paddle
arg_1_tensor = paddle.randint(-64,4096,[2, 6], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,16384,[2, 4], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
res = paddle.searchsorted(arg_1,arg_2,right=arg_3,)
