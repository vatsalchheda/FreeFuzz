import paddle
arg_1_tensor = paddle.randint(-2,256,[4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,64,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.elementwise_mod(arg_1,arg_2,)
