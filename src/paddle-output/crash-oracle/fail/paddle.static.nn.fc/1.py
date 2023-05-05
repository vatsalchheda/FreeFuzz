import paddle
arg_1_tensor = paddle.rand([-1, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 110
res = paddle.static.nn.fc(arg_1,arg_2,)
