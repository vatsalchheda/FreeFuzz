import paddle
arg_1_tensor = paddle.rand([2, 3, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = "NCHW"
arg_4 = None
res = paddle.nn.functional.adaptive_avg_pool2d(arg_1,output_size=arg_2,data_format=arg_3,name=arg_4,)
