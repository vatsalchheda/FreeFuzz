import paddle
arg_1_0 = 60
arg_1_1 = 1024
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
res = paddle.randn(shape=arg_1,dtype=arg_2,)
