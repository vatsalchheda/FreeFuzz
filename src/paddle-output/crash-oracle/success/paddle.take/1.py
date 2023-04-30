import paddle
arg_1_tensor = paddle.randint(-128,512,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,16,[1, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = "clip"
res = paddle.take(arg_1,arg_2,mode=arg_3,)
