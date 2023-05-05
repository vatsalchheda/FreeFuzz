import paddle
arg_1_tensor = paddle.randint(-8,16,[11, 4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,4096,[11, 4, 4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.gather_tree(arg_1,arg_2,)
