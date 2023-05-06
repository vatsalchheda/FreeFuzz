import paddle
arg_1_tensor = paddle.randint(-128, 4, [6], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 74
arg_3 = True
arg_4 = True
res = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
