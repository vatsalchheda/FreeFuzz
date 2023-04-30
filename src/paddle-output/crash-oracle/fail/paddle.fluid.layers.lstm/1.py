import paddle
arg_1_tensor = paddle.randint(-64,4096,[-1, 100, 256], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,512,[1, 100, 150], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-256,4,[1, 100, 150], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 12
arg_5 = 150
arg_6 = 1
arg_7 = 0.2
res = paddle.fluid.layers.lstm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,dropout_prob=arg_7,)
