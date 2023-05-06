import paddle
arg_1 = False
arg_2_0 = -1
arg_2_1 = True
arg_2_2 = "zeros"
arg_2_3 = "replicate"
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "float32"
res = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
