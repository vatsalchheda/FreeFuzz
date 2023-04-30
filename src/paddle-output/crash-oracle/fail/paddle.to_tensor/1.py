import paddle
arg_1_tensor = paddle.randint(-128,2048,[37574, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.to_tensor(arg_1,)
