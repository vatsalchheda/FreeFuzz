import paddle
arg_1_tensor = paddle.rand([32, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4, 8, [32, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = -81
arg_5 = "mean"
arg_6 = True
arg_7 = -1
arg_8 = True
arg_9 = None
res = paddle.nn.functional.cross_entropy(arg_1,arg_2,weight=arg_3,ignore_index=arg_4,reduction=arg_5,soft_label=arg_6,axis=arg_7,use_softmax=arg_8,name=arg_9,)
