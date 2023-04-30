import paddle
arg_1_tensor = paddle.randint(-4096,2,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.eig(arg_1,)
