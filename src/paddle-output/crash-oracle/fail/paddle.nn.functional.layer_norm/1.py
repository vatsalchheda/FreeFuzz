import paddle
arg_1_tensor = paddle.rand([1, 52800], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 8
res = paddle.nn.functional.layer_norm(arg_1,arg_2,)
