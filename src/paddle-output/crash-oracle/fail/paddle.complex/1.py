import paddle
arg_1_tensor = paddle.randint(-1024,2048,[4, 4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,2048,[4, 4, 4], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.complex(arg_1,arg_2,)
