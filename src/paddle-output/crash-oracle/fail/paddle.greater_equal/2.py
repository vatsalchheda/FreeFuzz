import paddle
arg_1_tensor = paddle.randint(-16,64,[1, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,8192,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.greater_equal(arg_1,arg_2,)
