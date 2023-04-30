import paddle
arg_1_tensor = paddle.randint(-8,64,[3, 1], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,1024,[3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = "mean"
arg_5 = None
arg_6 = None
res = paddle.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
