import paddle
arg_1_tensor = paddle.rand([2, 3, 8, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
res = paddle.nn.functional.conv3d(arg_1,arg_2,)
