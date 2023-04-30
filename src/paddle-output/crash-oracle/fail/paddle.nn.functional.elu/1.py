import paddle
arg_1_tensor = paddle.randint(-2048,32,[3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
res = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
