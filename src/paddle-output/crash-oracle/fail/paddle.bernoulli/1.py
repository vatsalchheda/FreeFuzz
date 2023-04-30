import paddle
arg_1_tensor = paddle.randint(-128,1024,[2, 16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.bernoulli(arg_1,)
