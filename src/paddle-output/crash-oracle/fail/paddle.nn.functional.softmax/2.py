import paddle
arg_1_tensor = paddle.rand([32, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -31
arg_3 = None
arg_4 = None
res = paddle.nn.functional.softmax(arg_1,arg_2,arg_3,arg_4,)
