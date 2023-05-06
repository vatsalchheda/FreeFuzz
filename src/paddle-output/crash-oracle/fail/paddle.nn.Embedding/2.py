import paddle
arg_1 = "max"
arg_2 = 8
arg_class = paddle.nn.Embedding(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-256, 8192, [1, 1], dtype=paddle.int64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
