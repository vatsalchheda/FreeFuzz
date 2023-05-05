import paddle
arg_class = paddle.nn.CrossEntropyLoss()
arg_1_0_tensor = paddle.rand([32, 10], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-1, 512, [32, 1], dtype=paddle.int64arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
