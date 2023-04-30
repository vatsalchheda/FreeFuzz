import paddle
arg_1_tensor = paddle.randint(-32768,16,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.logsumexp(arg_1,arg_2,)
