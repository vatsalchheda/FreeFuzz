import paddle
arg_1_tensor = paddle.randint(-64,16,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "The content of input layer:"
res = paddle.static.Print(arg_1,message=arg_2,)
