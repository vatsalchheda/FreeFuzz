import paddle
arg_1_tensor = paddle.randint(-1024,8192,[1, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,1024,[2, 3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.conv1d(arg_1,arg_2,)
