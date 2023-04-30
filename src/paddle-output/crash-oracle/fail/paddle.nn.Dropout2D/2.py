import paddle
arg_1 = 0.5
arg_class = paddle.nn.Dropout2D(p=arg_1,)
arg_2_0_tensor = paddle.randint(-4,512,[2, 2, 1, 3], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
