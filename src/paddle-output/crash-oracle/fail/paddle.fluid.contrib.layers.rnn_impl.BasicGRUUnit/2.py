import paddle
arg_1 = "basic_gru_reverse_layers_1"
arg_2 = False
arg_class = paddle.fluid.contrib.layers.rnn_impl.BasicGRUUnit(arg_1,arg_2,)
arg_3_0_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([-1, 256], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
