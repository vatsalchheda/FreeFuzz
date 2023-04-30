import paddle
arg_1_tensor = paddle.randint(-4096,128,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,256,[6, 3, 3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.conv2d(arg_1,arg_2,)
