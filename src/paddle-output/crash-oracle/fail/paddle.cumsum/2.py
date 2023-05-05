import paddle
arg_1_tensor = paddle.randint(-128, 128, [1, 45], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
res = paddle.cumsum(arg_1,axis=arg_2,)
