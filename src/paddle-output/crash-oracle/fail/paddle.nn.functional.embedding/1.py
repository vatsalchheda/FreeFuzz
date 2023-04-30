import paddle
arg_1_tensor = paddle.randint(-32768,32,[3, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,64,[10, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -16
arg_4 = "sum"
res = paddle.nn.functional.embedding(x=arg_1,weight=arg_2,sparse=arg_3,name=arg_4,)
