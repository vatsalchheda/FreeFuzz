import paddle
arg_1_0 = -25
arg_1_1 = 1
arg_1_2 = 42
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = "float32"
res = paddle.rand(arg_1,dtype=arg_2,)
