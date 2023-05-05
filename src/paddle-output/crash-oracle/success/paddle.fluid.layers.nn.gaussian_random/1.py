import paddle
arg_1_0 = 256
arg_1_1 = 256
arg_1_2 = 11
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = 18.0
arg_3 = 62.0
arg_4 = -44
arg_5 = "float32"
res = paddle.fluid.layers.nn.gaussian_random(arg_1,mean=arg_2,std=arg_3,seed=arg_4,dtype=arg_5,)
