import paddle
arg_1 = 1.0
arg_class = paddle.nn.Dropout(p=arg_1,)
arg_2_0_tensor = paddle.randint(-512,1024,[2, 3], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
