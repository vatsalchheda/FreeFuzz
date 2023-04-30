import paddle
arg_1_tensor = paddle.randint(-32768,16384,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = 1
res = paddle.clip(arg_1,arg_2,arg_3,)
