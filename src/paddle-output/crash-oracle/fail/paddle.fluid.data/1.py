import paddle
arg_1 = -35
arg_2_0 = 8
arg_2_1 = 20
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "int32"
res = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
