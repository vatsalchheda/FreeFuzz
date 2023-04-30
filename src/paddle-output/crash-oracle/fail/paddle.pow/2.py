import paddle
arg_1_tensor = paddle.randint(-2048,8,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 18.0
res = paddle.pow(arg_1,arg_2,)
