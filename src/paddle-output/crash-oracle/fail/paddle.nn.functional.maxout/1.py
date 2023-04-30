import paddle
arg_1_tensor = paddle.randint(-16384,256,[1, 2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 8
res = paddle.nn.functional.maxout(arg_1,groups=arg_2,)
