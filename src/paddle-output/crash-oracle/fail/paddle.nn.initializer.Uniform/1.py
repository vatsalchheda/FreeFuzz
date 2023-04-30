import paddle
arg_class = paddle.nn.initializer.Uniform()
arg_1_0_tensor = paddle.randint(-512,8192,[1024, 64], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32768,32768,[2, 2], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
