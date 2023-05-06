import paddle
arg_1_tensor = paddle.randint(-4096, 256, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
res = paddle.is_complex(arg_1,)
