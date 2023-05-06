import paddle
arg_1_tensor = paddle.rand([1, 30001], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = "float64"
res = paddle.cumprod(arg_1,dim=arg_2,dtype=arg_3,)
