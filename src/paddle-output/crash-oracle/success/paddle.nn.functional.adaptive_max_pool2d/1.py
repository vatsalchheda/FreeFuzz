import paddle
arg_1_tensor = paddle.rand([2, 3, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 21
arg_3 = True
arg_4 = 60
res = paddle.nn.functional.adaptive_max_pool2d(arg_1,output_size=arg_2,return_mask=arg_3,name=arg_4,)
