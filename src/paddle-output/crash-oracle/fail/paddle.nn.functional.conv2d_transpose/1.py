import paddle
arg_1_tensor = paddle.randint(-1024,2048,[2, 3, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,16,[3, 6, 3, 0], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.conv2d_transpose(arg_1,arg_2,)
