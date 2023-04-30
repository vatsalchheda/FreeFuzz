import paddle
arg_1_tensor = paddle.randint(-2,128,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,2048,[2, 2, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-256,512,[2, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = 5.0
res = paddle.addmm(input=arg_1,x=arg_2,y=arg_3,beta=arg_4,alpha=arg_5,)
