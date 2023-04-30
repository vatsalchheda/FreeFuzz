import paddle
arg_1_tensor = paddle.randint(-1,4096,[5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifftshift(arg_1,)
