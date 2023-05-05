import paddle
arg_1_tensor = paddle.randint(-1,256,[100], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,128,[200], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.meshgrid(arg_1,arg_2,)
