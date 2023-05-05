import paddle
arg_1_tensor = paddle.rand([1, 2, 46, 46], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.1
arg_3 = True
arg_4 = "channel"
res = paddle.nn.functional.dropout(arg_1,arg_2,training=arg_3,mode=arg_4,)
