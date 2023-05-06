import paddle
arg_1_tensor = paddle.randint(-2, 8, [3, 2, 2], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096, 32768, [3, 2, 2], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.gather_tree(arg_1,arg_2,)
