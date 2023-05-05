import paddle
arg_1_tensor = paddle.randint(-64,2048,[1, 15], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "float32"
res = paddle.zeros_like(arg_1,dtype=arg_2,)
