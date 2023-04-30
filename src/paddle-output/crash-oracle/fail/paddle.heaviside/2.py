import paddle
arg_1_tensor = paddle.randint(-256,4,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,4,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.heaviside(arg_1,arg_2,)
