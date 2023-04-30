import paddle
arg_1 = 51.5
arg_class = paddle.nn.Dropout(p=arg_1,)
arg_2_0_tensor = paddle.randint(-4096,8192,[2, 3], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
