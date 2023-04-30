import paddle
arg_1_tensor = paddle.randint(-128,4096,[-1, 6, 26, 26], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "relu"
res = paddle.static.nn.batch_norm(input=arg_1,act=arg_2,)
