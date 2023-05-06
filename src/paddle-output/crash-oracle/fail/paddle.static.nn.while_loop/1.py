import paddle
arg_1 = "cond"
arg_2 = "body"
arg_3_0_tensor = paddle.randint(-4096, 32768, [1], dtype=paddle.int64arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.static.nn.while_loop(cond=arg_1,body=arg_2,loop_vars=arg_3,)
