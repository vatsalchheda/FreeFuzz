import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 15
res = paddle.static.nn.fc(arg_1,size=arg_2,)
