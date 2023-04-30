import paddle
arg_class = paddle.nn.BCEWithLogitsLoss()
arg_1_0_tensor = paddle.randint(-2048,512,[41], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(0,2,[], dtype=paddle.bool)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
