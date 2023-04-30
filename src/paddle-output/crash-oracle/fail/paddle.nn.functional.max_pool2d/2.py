import paddle
arg_1_tensor = paddle.randint(-16,32768,[-1, 20, 24, 24], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -18
arg_3 = 2
res = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,)
