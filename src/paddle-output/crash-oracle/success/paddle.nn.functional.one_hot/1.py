import paddle
arg_1_tensor = paddle.randint(-2048,32,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
res = paddle.nn.functional.one_hot(arg_1,num_classes=arg_2,)
