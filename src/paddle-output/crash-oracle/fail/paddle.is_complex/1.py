import paddle
arg_1_tensor = paddle.randint(-2048,1,[2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.is_complex(arg_1,)
