import paddle
arg_1_tensor = paddle.randint(-4,4096,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.sort(arg_1,arg_2,)
