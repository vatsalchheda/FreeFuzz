import paddle
arg_1_tensor = paddle.rand([1, 512, 62, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = 35.0
arg_4 = None
res = paddle.nn.functional.channel_shuffle(arg_1,arg_2,arg_3,arg_4,)
