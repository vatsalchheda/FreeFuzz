import paddle
arg_1 = 2
arg_class = paddle.nn.Maxout(groups=arg_1,)
arg_2_0_tensor = paddle.randint(-2048,2,[1, 2, 3, 4], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
