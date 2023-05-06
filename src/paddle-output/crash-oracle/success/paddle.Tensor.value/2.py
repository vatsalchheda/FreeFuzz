import paddle
arg_1_tensor = paddle.randint(0,2,[3])
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.value(arg_1,)
