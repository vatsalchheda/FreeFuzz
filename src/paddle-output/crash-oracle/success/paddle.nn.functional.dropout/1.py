import paddle
arg_1_tensor = paddle.rand([1, 1, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.1
arg_3 = None
arg_4 = False
arg_5 = "upscale_in_train"
arg_6 = None
res = paddle.nn.functional.dropout(arg_1,p=arg_2,axis=arg_3,training=arg_4,mode=arg_5,name=arg_6,)
