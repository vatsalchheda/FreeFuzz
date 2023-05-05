import paddle
arg_1_tensor = paddle.randint(-128,128,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -38
arg_3 = "paddleVarType"
res = paddle.fluid.layers.sequence_lod.sequence_mask(arg_1,maxlen=arg_2,dtype=arg_3,)
