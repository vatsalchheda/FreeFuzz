import paddle
arg_1_tensor = paddle.randint(-4,4096,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.multi_dot(arg_1,)
