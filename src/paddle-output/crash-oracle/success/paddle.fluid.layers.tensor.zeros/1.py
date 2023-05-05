import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
res = paddle.fluid.layers.tensor.zeros(arg_1,dtype=arg_2,)
