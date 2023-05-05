import paddle
arg_1_tensor = paddle.randint(-8192, 2048, [4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32, 32768, [1], dtype=paddle.int32arg_2 = arg_2_tensor.clone()
arg_3 = "paddleVarType"
res = paddle.fluid.layers.sequence_lod.sequence_mask(arg_1,maxlen=arg_2,dtype=arg_3,)
