import paddle
arg_1_tensor = paddle.rand([2, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8, 2048, [1], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.control_flow.less_equal(arg_1,arg_2,)
