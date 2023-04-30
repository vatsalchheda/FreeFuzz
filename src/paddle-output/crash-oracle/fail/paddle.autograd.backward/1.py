import paddle
arg_1_0_tensor = paddle.randint(-16,64,[2, 0, 1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-256,4096,[2, 12, 1], dtype=paddle.bfloat16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.autograd.backward(arg_1,)
