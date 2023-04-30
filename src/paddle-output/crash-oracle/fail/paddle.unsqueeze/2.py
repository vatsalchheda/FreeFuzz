import paddle
arg_1_tensor = paddle.randint(-1,16,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.unsqueeze(arg_1,arg_2,)
