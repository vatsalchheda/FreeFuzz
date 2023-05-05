import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -39
res = paddle.fluid.layers.tensor.argmax(arg_1,axis=arg_2,)
