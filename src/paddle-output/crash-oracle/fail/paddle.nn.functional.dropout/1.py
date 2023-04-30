import paddle
arg_1_tensor = paddle.randint(-16384,1,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 27
res = paddle.nn.functional.dropout(arg_1,axis=arg_2,)
