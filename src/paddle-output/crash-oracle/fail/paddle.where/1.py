import paddle
arg_1_tensor = paddle.randint(0,2,[1, 30001])
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1, 30001], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(0,2,[2, 2])
arg_3 = arg_3_tensor.clone()
res = paddle.where(arg_1,arg_2,arg_3,)
