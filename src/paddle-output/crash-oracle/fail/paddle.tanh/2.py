import paddle
arg_1_tensor = paddle.randint(-4096,512,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.tanh(arg_1,)
