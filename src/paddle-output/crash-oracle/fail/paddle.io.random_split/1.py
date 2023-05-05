import paddle
arg_1 = -36.0
arg_2_0 = -60.0
arg_2_1 = "zeros"
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.io.random_split(arg_1,arg_2,)
