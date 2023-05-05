import paddle
arg_1_tensor = paddle.randint(-16,2,[4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,2,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.control_flow.equal(arg_1,arg_2,)
