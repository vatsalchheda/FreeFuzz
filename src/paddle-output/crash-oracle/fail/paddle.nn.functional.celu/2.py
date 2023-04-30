import paddle
arg_1_tensor = paddle.randint(-1,2048,[1, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.2
arg_3 = None
res = paddle.nn.functional.celu(arg_1,arg_2,arg_3,)
