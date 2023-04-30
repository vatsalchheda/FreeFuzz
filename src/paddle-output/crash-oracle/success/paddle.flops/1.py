import paddle
arg_1 = "__main__LeNet"
arg_2_0 = -30.0
arg_2_1 = False
arg_2_2 = False
arg_2_3 = -54.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "mean"
arg_4 = 1e+20
res = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
