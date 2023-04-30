import paddle
arg_1_tensor = paddle.randint(-2048,4096,[2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.amax(arg_1,axis=arg_2,)
