import paddle
arg_1_tensor = paddle.randint(-4096,128,[1, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,512,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.index_sample(arg_1,arg_2,)
