import paddle
arg_1_tensor = paddle.rand([-1, 3, 6, 9], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "BILINEAR"
res = paddle.fluid.layers.nn.image_resize(input=arg_1,out_shape=arg_2,resample=arg_3,)
