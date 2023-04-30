import paddle
arg_1_tensor = paddle.randint(-4,4096,[6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
arg_3 = True
res = paddle.sum(arg_1,axis=arg_2,keepdim=arg_3,)
