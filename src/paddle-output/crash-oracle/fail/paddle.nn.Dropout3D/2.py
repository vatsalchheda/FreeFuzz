import paddle
arg_1 = "replicate"
arg_class = paddle.nn.Dropout3D(p=arg_1,)
arg_2_0_tensor = paddle.randint(-1024,32,[1, 2, 2, 2, 3], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
