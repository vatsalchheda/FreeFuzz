import paddle
arg_1_tensor = paddle.randint(-256, 512, [22, 4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -58
arg_3 = "max"
res = paddle.median(arg_1,axis=arg_2,keepdim=arg_3,)
