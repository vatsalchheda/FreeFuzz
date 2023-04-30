import paddle
arg_1_tensor = paddle.randint(-64,16,[3, 4, 5, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = False
res = paddle.sum(arg_1,axis=arg_2,keepdim=arg_3,)
