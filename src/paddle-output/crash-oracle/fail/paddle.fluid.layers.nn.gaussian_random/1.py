import paddle
arg_1_0 = 1
arg_1_1 = 1024
arg_1_2 = -1024
arg_1_3 = 60
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = 0.0
arg_3 = 1.0
arg_4 = True
arg_5 = "float32"
res = paddle.fluid.layers.nn.gaussian_random(arg_1,mean=arg_2,std=arg_3,seed=arg_4,dtype=arg_5,)
