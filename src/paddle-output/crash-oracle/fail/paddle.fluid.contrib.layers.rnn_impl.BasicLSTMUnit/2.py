import paddle
arg_1 = "max"
arg_2 = 256
arg_3 = None
arg_4 = None
arg_5 = None
arg_6 = "zeros"
arg_7 = 1.0
arg_8 = "float32"
arg_class = paddle.fluid.contrib.layers.rnn_impl.BasicLSTMUnit(arg_1,arg_2,param_attr=arg_3,bias_attr=arg_4,gate_activation=arg_5,activation=arg_6,forget_bias=arg_7,dtype=arg_8,)
arg_9_0_tensor = paddle.rand([20, 256], dtype=paddle.float32)
arg_9_0 = arg_9_0_tensor.clone()
arg_9_1_tensor = paddle.rand([-1, 256], dtype=paddle.float32)
arg_9_1 = arg_9_1_tensor.clone()
arg_9_2_tensor = paddle.rand([-1, 256], dtype=paddle.float32)
arg_9_2 = arg_9_2_tensor.clone()
arg_9 = [arg_9_0,arg_9_1,arg_9_2,]
res = arg_class(*arg_9)
