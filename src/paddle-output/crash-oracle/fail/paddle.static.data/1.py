import paddle
arg_1 = "click"
arg_2_0 = 45
arg_2 = [arg_2_0,]
arg_3 = "int64"
res = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,)
