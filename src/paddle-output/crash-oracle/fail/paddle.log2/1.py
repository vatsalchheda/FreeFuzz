import paddle
arg_1_tensor = paddle.randint(-1,512,[2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.log2(arg_1,)
