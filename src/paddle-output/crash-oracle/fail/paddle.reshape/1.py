import paddle
arg_1_tensor = paddle.randint(-128, 4096, [64, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0 = -56.0
arg_2_1 = False
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.reshape(arg_1,arg_2,)
