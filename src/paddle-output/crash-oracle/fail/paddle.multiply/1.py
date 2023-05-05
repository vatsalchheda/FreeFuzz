import paddle
arg_1_tensor = paddle.rand([38, 512, 123], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[2, 2])
arg_2 = arg_2_tensor.clone()
arg_3 = None
res = paddle.multiply(arg_1,arg_2,name=arg_3,)
