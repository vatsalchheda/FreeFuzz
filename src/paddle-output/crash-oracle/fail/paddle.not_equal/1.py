import paddle
arg_1_tensor = paddle.randint(-64,512,[7, 8], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,32768,[7, 8], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.not_equal(arg_1,arg_2,)
