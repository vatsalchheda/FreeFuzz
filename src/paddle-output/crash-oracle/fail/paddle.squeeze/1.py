import paddle
arg_1_tensor = paddle.randint(-128,1024,[1, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.squeeze(arg_1,axis=arg_2,)
