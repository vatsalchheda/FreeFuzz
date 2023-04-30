import paddle
arg_1_tensor = paddle.randint(-256,64,[513], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.sin(arg_1,)
