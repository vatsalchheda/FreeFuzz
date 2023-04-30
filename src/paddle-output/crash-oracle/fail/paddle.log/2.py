import paddle
arg_1_tensor = paddle.randint(-16384,128,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "Categorical_log_prob"
res = paddle.log(arg_1,name=arg_2,)
