import paddle
arg_1_tensor = paddle.randint(-1024,4,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.sign(x=arg_1,)
