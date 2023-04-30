import paddle
arg_1_tensor = paddle.randint(-32,4096,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.corrcoef(arg_1,)
