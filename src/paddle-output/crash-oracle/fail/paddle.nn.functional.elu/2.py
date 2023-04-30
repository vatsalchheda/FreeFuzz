import paddle
arg_1_tensor = paddle.randint(-4096,8192,[3, 1, 7, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.2
res = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
