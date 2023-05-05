import paddle
arg_1 = 1.5
arg_2 = "upscale_in_train"
arg_class = paddle.nn.Dropout(arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.rand([4, 1, 8], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
