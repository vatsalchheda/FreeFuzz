import paddle
arg_1_tensor = paddle.rand([-1, 2048], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2048
arg_3 = False
res = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
