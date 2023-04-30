import paddle
arg_1_tensor = paddle.randint(-256,1024,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,256,[2, 1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-8,32768,[2, 1], dtype=paddle.float64)
arg_3 = arg_3_tensor.clone()
res = paddle.lerp(arg_1,arg_2,arg_3,)
