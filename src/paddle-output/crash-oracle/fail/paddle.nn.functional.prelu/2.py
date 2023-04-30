import paddle
arg_1_tensor = paddle.randint(-1024,8192,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,2,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.prelu(arg_1,arg_2,)
