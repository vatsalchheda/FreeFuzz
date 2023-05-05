import paddle
arg_1 = 5
arg_2 = 4
arg_3 = True
arg_class = paddle.fluid.dygraph.nn.BilinearTensorProduct(input1_dim=arg_1,input2_dim=arg_2,output_dim=arg_3,)
arg_4_0_tensor = paddle.rand([5, 5], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([5, 4], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
