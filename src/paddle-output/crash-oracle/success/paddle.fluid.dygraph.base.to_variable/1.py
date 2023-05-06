import paddle
arg_1_0 = 1e-06
arg_1 = [arg_1_0,]
arg_2 = "float32"
res = paddle.fluid.dygraph.base.to_variable(arg_1,dtype=arg_2,)
