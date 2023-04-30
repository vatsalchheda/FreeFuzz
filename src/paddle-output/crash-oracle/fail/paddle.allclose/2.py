import paddle
arg_1_tensor = paddle.randint(-8192,16,[1, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,32768,[1, 8], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.allclose(arg_1,arg_2,)
