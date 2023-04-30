import paddle
arg_1_tensor = paddle.randint(-256,8192,[3, 2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,16384,[3, 2, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.gather_tree(arg_1,arg_2,)
