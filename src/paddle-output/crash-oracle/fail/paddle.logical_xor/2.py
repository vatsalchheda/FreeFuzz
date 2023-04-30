import paddle
arg_1_tensor = paddle.randint(-16,512,[2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[2, 2], dtype=paddle.bool)
arg_2 = arg_2_tensor.clone()
res = paddle.logical_xor(arg_1,arg_2,)
