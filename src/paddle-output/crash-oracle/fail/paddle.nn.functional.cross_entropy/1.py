import paddle
arg_1_tensor = paddle.rand([64, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512, 32768, [64, 1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = -158
arg_5 = "sum"
arg_6 = False
arg_7 = 120.0
arg_8 = True
arg_9 = -972.0
res = paddle.nn.functional.cross_entropy(arg_1,arg_2,weight=arg_3,ignore_index=arg_4,reduction=arg_5,soft_label=arg_6,axis=arg_7,use_softmax=arg_8,name=arg_9,)
