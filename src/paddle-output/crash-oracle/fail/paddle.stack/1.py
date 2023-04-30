import paddle
arg_1_0_tensor = paddle.randint(-4,256,[1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-2,32768,[1], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = False
res = paddle.stack(arg_1,arg_2,)
