import paddle
arg_1_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
res = paddle.nn.functional.relu(arg_1,arg_2,)
