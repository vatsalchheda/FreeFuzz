import paddle
arg_1 = -47
arg_2 = 2
arg_3 = 0
arg_4 = True
arg_class = paddle.nn.MaxPool2D(kernel_size=arg_1,stride=arg_2,padding=arg_3,return_mask=arg_4,)
arg_5_0_tensor = paddle.randint(-128,4,[1, 3, 32, 32], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
res = arg_class(*arg_5)
