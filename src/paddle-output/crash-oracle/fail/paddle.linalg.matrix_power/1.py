import paddle
arg_1_tensor = paddle.randint(-16384,128,[3, 4, 5, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.linalg.matrix_power(arg_1,arg_2,)
