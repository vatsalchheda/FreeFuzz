import paddle
arg_1_tensor = paddle.randint(-8192,1024,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,1,[1, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 83
arg_4 = 0
res = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
