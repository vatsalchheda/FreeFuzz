import paddle
arg_class = paddle.nn.NLLLoss()
arg_1_0_tensor = paddle.rand([5, 3], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-1,64,[5], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
