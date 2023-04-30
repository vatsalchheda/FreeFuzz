import paddle
arg_1_tensor = paddle.randint(-512,64,[-1, 3, 6, 8, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 45
arg_2_1 = 71
arg_2_2 = -49
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = None
arg_4 = None
arg_5 = "TRILINEAR"
arg_6 = None
arg_7 = -38.0
arg_8 = 1
arg_9 = "NCDHW"
res = paddle.fluid.layers.nn.image_resize(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,)
