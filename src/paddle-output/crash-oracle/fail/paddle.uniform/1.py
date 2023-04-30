import paddle
arg_1_0 = 1
arg_1_1 = 2
arg_1_2 = 1
arg_1_3 = 1
arg_1_4 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
arg_2 = "replicate"
arg_3 = 0.0
arg_4 = 3
res = paddle.uniform(arg_1,dtype=arg_2,min=arg_3,max=arg_4,)
