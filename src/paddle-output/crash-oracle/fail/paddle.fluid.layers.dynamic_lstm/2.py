import paddle
arg_1_tensor = paddle.rand([16, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1024
arg_2_1 = 64
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 50.0
res = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
