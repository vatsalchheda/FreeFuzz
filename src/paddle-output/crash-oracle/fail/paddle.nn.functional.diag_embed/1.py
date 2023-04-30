import paddle
arg_1_tensor = paddle.randint(-8192,4,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.diag_embed(arg_1,)
