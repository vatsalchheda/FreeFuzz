import paddle
arg_1_tensor = paddle.randint(-16384,128,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,1024,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.autograd.grad(arg_1,arg_2,)
