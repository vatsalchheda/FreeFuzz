import paddle
arg_1_tensor = paddle.randint(-8, 8192, [2, 4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16, 1024, [41], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 20.0
res = paddle.bucketize(arg_1,arg_2,right=arg_3,)
