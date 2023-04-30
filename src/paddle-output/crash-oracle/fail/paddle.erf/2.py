import paddle
arg_1_tensor = paddle.randint(0,64,[4, 1], dtype=paddle.uint8)
arg_1 = arg_1_tensor.clone()
res = paddle.erf(arg_1,)
