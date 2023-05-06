import paddle
arg_1_tensor = paddle.randint(0,2,[64, 1])
arg_1 = arg_1_tensor.clone()
arg_2 = "float32"
res = paddle.cast(arg_1,dtype=arg_2,)
