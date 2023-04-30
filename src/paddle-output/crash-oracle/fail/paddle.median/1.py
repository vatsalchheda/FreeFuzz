import paddle
arg_1_tensor = paddle.randint(-256,4,[3, 55], dtype=paddle.int16)
arg_1 = arg_1_tensor.clone()
res = paddle.median(arg_1,)
