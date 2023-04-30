import paddle
arg_1_tensor = paddle.randint(-512,2,[2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
res = paddle.mode(arg_1,arg_2,)
