import paddle
arg_1 = True
arg_2 = "max"
arg_3 = 1
arg_4 = False
arg_class = paddle.fluid.dygraph.nn.Pool2D(pool_size=arg_1,pool_type=arg_2,pool_stride=arg_3,global_pooling=arg_4,)
arg_5_0_tensor = paddle.rand([3, 32, 32, 5], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
res = arg_class(*arg_5)
