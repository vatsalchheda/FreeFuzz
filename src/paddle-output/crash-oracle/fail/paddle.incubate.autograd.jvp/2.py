import paddle
arg_1 = "func"
arg_2_tensor = paddle.randint(-32,2048,[2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-128,16384,[2, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.incubate.autograd.jvp(arg_1,arg_2,arg_3,)
