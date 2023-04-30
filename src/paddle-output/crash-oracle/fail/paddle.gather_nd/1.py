import paddle
arg_1_tensor = paddle.randint(-16,16,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,32768,[1, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gather_nd(arg_1,arg_2,)
