import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "The content of input layer:"
res = paddle.static.Print(arg_1,message=arg_2,)
