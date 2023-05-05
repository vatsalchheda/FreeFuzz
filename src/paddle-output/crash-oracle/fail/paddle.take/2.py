import paddle
arg_1_tensor = paddle.randint(-4096, 512, [3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 256, [3, 5], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = False
res = paddle.take(arg_1,arg_2,mode=arg_3,)
