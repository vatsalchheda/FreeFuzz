import paddle
arg_1_tensor = paddle.randint(-128,16,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 0
arg_4 = False
res = paddle.topk(arg_1,arg_2,axis=arg_3,largest=arg_4,)
