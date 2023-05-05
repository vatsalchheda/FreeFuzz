import paddle
arg_1_tensor = paddle.rand([-1, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = "translated_layer/scale_0"
res = paddle.fluid.layers.nn.scale(arg_1,arg_2,name=arg_3,)
