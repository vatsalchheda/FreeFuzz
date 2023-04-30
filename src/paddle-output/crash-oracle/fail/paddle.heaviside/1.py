import paddle
arg_1_tensor = paddle.randint(-4096,16,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,1,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.heaviside(arg_1,arg_2,)
