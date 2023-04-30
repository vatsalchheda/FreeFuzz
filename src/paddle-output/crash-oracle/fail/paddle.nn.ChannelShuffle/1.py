import paddle
arg_1 = 3
arg_class = paddle.nn.ChannelShuffle(arg_1,)
arg_2_0_tensor = paddle.randint(-512,32768,[1, 6, 1, 1], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
