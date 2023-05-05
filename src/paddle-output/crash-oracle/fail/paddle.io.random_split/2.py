import paddle
arg_1 = "builtinsrange"
arg_2_0 = 15
arg_2_1 = 53
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.io.random_split(arg_1,arg_2,)
