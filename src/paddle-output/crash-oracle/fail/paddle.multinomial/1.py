import paddle
arg_1_tensor = paddle.randint(-2,8,[2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
res = paddle.multinomial(arg_1,num_samples=arg_2,)
