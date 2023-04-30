import paddle
arg_1_tensor = paddle.randint(-4,16,[4, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,1024,[0, 3], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = True
arg_4 = -1
arg_5 = None
arg_6 = "mean"
res = paddle.nn.functional.cross_entropy(arg_1,arg_2,soft_label=arg_3,axis=arg_4,weight=arg_5,reduction=arg_6,)
