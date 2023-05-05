import paddle
arg_1 = 0.0
arg_2_0 = 256
arg_2_1 = 256
arg_2_2 = 11
arg_2_3 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "float32"
arg_4 = False
arg_5 = None
res = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
