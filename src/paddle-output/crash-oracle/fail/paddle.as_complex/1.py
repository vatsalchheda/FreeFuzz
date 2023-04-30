import paddle
arg_1_tensor = paddle.randint(-64,8,[2, 3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.as_complex(arg_1,)
