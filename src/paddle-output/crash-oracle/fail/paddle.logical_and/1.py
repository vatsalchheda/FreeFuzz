import paddle
arg_1_tensor = paddle.randint(0,2,[1], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[4], dtype=paddle.bool)
arg_2 = arg_2_tensor.clone()
res = paddle.logical_and(arg_1,arg_2,)
