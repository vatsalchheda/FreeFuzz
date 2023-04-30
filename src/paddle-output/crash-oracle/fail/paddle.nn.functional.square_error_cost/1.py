import paddle
arg_1_tensor = paddle.randint(-8,1024,[31, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,2,[2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.square_error_cost(arg_1,arg_2,)
