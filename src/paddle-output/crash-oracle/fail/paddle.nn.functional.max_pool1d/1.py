import paddle
arg_1_tensor = paddle.randint(-512,1,[1, 3, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = "max"
arg_4 = 0
arg_5 = True
res = paddle.nn.functional.max_pool1d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,)
