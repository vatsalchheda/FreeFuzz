import paddle
arg_1_tensor = paddle.randint(-2048,1024,[5, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,32,[5, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.logical_and(arg_1,arg_2,)
