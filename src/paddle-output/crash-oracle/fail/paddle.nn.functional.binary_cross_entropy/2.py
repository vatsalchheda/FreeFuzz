import paddle
arg_1_tensor = paddle.randint(-8192,256,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,8,[3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.binary_cross_entropy(arg_1,arg_2,)
