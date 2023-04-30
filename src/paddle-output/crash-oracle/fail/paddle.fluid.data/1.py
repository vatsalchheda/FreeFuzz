import paddle
arg_1 = "score_1"
arg_2_0 = None
arg_2_1 = -34
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "float32"
arg_4 = 1
res = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,lod_level=arg_4,)
