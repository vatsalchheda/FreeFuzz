import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 0
res = paddle.fluid.layers.nn.uniform_random(arg_1,seed=arg_2,)
