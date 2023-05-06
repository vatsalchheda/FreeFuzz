import paddle
arg_1_tensor = paddle.rand([6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256, 8192, [3], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = "Categorical_probs"
res = paddle.gather(arg_1,arg_2,name=arg_3,)
