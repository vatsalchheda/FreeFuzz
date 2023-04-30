import paddle
arg_1_tensor = paddle.randint(-32768,16384,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 73
arg_3 = None
res = paddle.nn.functional.celu(arg_1,arg_2,arg_3,)
