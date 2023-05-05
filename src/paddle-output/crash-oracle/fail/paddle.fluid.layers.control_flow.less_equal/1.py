import paddle
arg_1_tensor = paddle.randint(-32,2048,[-1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,32,[1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.control_flow.less_equal(arg_1,arg_2,)
