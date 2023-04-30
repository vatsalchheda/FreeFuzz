import paddle
arg_1_tensor = paddle.randint(-4096,16384,[257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,2,[257], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.maximum(arg_1,arg_2,)
