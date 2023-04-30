import paddle
arg_1_tensor = paddle.randint(-8,64,[5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.shape(arg_1,)
