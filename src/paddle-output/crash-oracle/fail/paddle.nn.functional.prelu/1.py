import paddle
arg_1_tensor = paddle.randint(-2048,64,[2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -51.0
res = paddle.nn.functional.prelu(arg_1,arg_2,data_format=arg_3,)
