import paddle
arg_1_tensor = paddle.randint(-32,2,[1, 257, 274], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2.0
res = paddle.pow(arg_1,arg_2,)
