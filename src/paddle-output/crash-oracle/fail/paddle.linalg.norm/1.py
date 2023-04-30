import paddle
arg_1_tensor = paddle.randint(-64,2,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = inf
res = paddle.linalg.norm(arg_1,p=arg_2,)
