import paddle
arg_1_tensor = paddle.randint(-512,1024,[1, 2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,2048,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fmin(arg_1,arg_2,)
