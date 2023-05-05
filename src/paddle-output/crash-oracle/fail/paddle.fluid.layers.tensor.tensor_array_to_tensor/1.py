import paddle
arg_1_tensor = paddle.randint(-64, 64, [-1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -37
arg_3 = True
res = paddle.fluid.layers.tensor.tensor_array_to_tensor(arg_1,axis=arg_2,use_stack=arg_3,)
