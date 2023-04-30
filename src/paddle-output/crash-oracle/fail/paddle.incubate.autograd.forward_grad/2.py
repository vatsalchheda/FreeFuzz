import paddle
arg_1_tensor = paddle.randint(-256,2048,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,32,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.autograd.forward_grad(arg_1,arg_2,)
