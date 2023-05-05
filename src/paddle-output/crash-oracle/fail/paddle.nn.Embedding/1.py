import paddle
arg_1 = 978
arg_2 = 8
arg_class = paddle.nn.Embedding(arg_1,arg_2,)
arg_3_0_tensor = paddle.randint(-1024,1,[2], dtype=paddle.int32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-4,4096,[1], dtype=paddle.int32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
