import paddle
arg_1_tensor = paddle.randint(-8,2048,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = False
res = paddle.nn.functional.dropout(arg_1,arg_2,training=arg_3,)
