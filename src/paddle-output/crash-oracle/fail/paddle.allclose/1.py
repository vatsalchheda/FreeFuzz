import paddle
arg_1_tensor = paddle.randint(-1,1024,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,16,[2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1e-05
arg_4 = 21.00000001
arg_5 = True
arg_6 = "ignore_nan"
res = paddle.allclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
