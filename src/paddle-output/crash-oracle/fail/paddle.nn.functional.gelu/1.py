import paddle
arg_1_tensor = paddle.randint(-1,32768,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = None
res = paddle.nn.functional.gelu(arg_1,arg_2,arg_3,)
