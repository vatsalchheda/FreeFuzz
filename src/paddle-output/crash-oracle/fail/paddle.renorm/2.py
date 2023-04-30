import paddle
arg_1_tensor = paddle.randint(-512,2,[2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = -1024
arg_4 = 2.05
res = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
