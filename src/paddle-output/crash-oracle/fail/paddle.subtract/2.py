import paddle
arg_1_tensor = paddle.randint(-256,4096,[0, 2, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,128,[2, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.subtract(arg_1,arg_2,)
