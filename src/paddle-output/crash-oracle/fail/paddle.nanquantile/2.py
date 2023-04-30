import paddle
arg_1_tensor = paddle.randint(-8192,4,[2, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = 1
res = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,)
