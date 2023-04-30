import paddle
arg_1_tensor = paddle.randint(-8,2,[2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.unsqueeze(arg_1,axis=arg_2,)
