import paddle
arg_1_tensor = paddle.randint(-64,4,[64, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.argsort(arg_1,descending=arg_2,)
