import paddle
arg_1_tensor = paddle.randint(-8192, 16, [4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = "NCDHW"
arg_4 = None
res = paddle.nn.functional.adaptive_avg_pool3d(arg_1,output_size=arg_2,data_format=arg_3,name=arg_4,)
