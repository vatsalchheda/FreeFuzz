import paddle
arg_1 = 44.0
arg_2 = -46.98
arg_3_0 = "max"
arg_3_1 = -64.0
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
