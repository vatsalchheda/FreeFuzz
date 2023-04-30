import paddle
arg_1_tensor = paddle.randint(-64,512,[2, 2, 3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,32768,[2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.bmm(arg_1,arg_2,)
