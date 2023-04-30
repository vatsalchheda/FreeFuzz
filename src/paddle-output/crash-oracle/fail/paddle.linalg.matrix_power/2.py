import paddle
arg_1_tensor = paddle.randint(-1,128,[2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
res = paddle.linalg.matrix_power(arg_1,arg_2,)
