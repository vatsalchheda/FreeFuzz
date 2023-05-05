import paddle
arg_1_tensor = paddle.rand([2, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 52800
arg_2 = [arg_2_0,]
res = paddle.nn.functional.layer_norm(arg_1,arg_2,)
