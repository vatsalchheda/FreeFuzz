import paddle
arg_class = paddle.nn.Layer()
arg_1_0_tensor = paddle.randint(-4096,4096,[1, 1, 2, 2, 3], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-4,8192,[1, 1, 2, 0, 3], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
