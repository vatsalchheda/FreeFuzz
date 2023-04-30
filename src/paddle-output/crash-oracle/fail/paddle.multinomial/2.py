import paddle
arg_1_tensor = paddle.randint(-32,64,[1, 30000], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.multinomial(arg_1,)
