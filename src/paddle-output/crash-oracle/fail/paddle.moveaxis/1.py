import paddle
arg_1_tensor = paddle.randint(-32768,8,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -44
arg_3 = 1
res = paddle.moveaxis(arg_1,arg_2,arg_3,)
