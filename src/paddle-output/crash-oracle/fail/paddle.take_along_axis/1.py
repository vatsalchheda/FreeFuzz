import paddle
arg_1_tensor = paddle.randint(-4096,256,[4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,512,[4, 1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 34
res = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
