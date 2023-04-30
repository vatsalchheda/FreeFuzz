import paddle
arg_1_tensor = paddle.randint(-2,8192,[2, 3, 8, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,2048,[6, 3, 3, 3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.conv3d(arg_1,arg_2,)
