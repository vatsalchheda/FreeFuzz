import paddle
arg_1_0 = 128
arg_1_1 = 128
arg_1_2 = 3
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "float32"
res = paddle.fluid.layers.tensor.zeros(arg_1,dtype=arg_2,)
