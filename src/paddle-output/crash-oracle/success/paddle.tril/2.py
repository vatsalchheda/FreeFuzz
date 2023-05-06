import paddle
arg_1_tensor = paddle.randint(0,2,[22, 22])
arg_1 = arg_1_tensor.clone()
res = paddle.tril(arg_1,)
