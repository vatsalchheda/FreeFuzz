import paddle
arg_class = paddle.no_grad()
arg_1_0_tensor = paddle.randint(-512,8192,[2, 2], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = arg_class(*arg_1)
