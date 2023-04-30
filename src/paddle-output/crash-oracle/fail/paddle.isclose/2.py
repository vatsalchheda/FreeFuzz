import paddle
arg_1_tensor = paddle.randint(-2,128,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,8192,[2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1e-05
arg_4 = 46.00000001
arg_5 = False
arg_6 = "ignore_nan"
res = paddle.isclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
