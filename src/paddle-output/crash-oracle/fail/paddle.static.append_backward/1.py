import paddle
arg_1_tensor = paddle.rand([1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = "embedding_0.w_0"
arg_2_1 = "my_fc.w_0"
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,)
