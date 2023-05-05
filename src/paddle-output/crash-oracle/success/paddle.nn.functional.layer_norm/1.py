import paddle
arg_1_tensor = paddle.rand([1, 164, 1024], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 164
arg_2_1 = 1024
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.nn.functional.layer_norm(arg_1,arg_2,)
