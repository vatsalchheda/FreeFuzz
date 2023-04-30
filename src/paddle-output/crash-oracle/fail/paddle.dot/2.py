import paddle
arg_1_tensor = paddle.randint(-4096,16,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,2048,[2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.dot(arg_1,arg_2,)
