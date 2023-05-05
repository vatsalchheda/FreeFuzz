import paddle
arg_1_0 = -22
arg_1_1 = -47
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "paddleVarType"
res = paddle.fluid.layers.tensor.ones(shape=arg_1,dtype=arg_2,)
