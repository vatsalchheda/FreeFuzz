import paddle
arg_1_tensor = paddle.randint(-4,32,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.linalg.cond(arg_1,p=arg_2,)
