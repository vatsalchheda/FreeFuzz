import paddle
arg_1_0_tensor = paddle.randint(-32,512,[1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-64,1024,[1], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 8
res = paddle.stack(arg_1,arg_2,)
