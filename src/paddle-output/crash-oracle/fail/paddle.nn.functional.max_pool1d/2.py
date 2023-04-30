import paddle
arg_1_tensor = paddle.randint(-8192,4096,[1, 3, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3 = 24
arg_4 = 1
res = paddle.nn.functional.max_pool1d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,)
