import paddle
arg_1_tensor = paddle.rand([100, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256, 128, [10], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.gather(arg_1,arg_2,)
