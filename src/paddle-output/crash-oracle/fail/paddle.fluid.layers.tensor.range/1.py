import paddle
arg_1 = 6
arg_2_tensor = paddle.randint(-32768,64,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4 = "paddleVarType"
res = paddle.fluid.layers.tensor.range(arg_1,arg_2,arg_3,dtype=arg_4,)
