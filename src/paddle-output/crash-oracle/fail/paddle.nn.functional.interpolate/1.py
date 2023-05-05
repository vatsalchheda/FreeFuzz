import paddle
arg_1_tensor = paddle.rand([1, 1, 80, 137], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 40
arg_2_1 = -7
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "nearest"
res = paddle.nn.functional.interpolate(arg_1,scale_factor=arg_2,mode=arg_3,)
