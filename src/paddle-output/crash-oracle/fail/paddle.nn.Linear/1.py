import paddle
arg_1 = "max"
arg_2 = 1024
arg_class = paddle.nn.Linear(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-4096,16384,[1, 13], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
