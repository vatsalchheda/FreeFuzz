import paddle
arg_1 = "pre_h"
arg_2_0 = 4
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "float64"
res = paddle.static.data(arg_1,arg_2,dtype=arg_3,)
