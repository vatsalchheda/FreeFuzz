import paddle
arg_1_tensor = paddle.randint(-16384,8192,[2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
res = paddle.nn.functional.log_softmax(arg_1,dtype=arg_2,)
