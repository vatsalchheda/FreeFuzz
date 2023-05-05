import paddle
arg_1_tensor = paddle.rand([0, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 512], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([-1, 512], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.fluid.layers.lstm_unit(x_t=arg_1,hidden_t_prev=arg_2,cell_t_prev=arg_3,)
