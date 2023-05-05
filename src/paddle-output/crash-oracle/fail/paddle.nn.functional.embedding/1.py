import paddle
arg_1_tensor = paddle.randint(-128,32768,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1026, 8], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = True
arg_5 = None
res = paddle.nn.functional.embedding(arg_1,weight=arg_2,padding_idx=arg_3,sparse=arg_4,name=arg_5,)
