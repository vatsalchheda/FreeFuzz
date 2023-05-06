import paddle
arg_1_tensor = paddle.rand([1, 4, 20, 20], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = False
arg_4 = "wrap"
res = paddle.nn.functional.dropout(arg_1,arg_2,training=arg_3,mode=arg_4,)
