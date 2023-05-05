import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 6
arg_3 = 3
res = paddle.static.nn.conv2d(input=arg_1,num_filters=arg_2,filter_size=arg_3,)
