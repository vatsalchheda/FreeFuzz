import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -42
arg_3 = None
res = paddle.nn.functional.thresholded_relu(arg_1,arg_2,arg_3,)
