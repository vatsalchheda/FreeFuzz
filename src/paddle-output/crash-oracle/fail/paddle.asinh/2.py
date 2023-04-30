import paddle
arg_1_tensor = paddle.randint(-128,32,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.asinh(arg_1,)
