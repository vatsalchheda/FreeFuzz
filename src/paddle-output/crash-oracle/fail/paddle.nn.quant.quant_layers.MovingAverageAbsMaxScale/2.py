import paddle
arg_1 = 49.0
arg_2 = "paddleVarType"
arg_class = paddle.nn.quant.quant_layers.MovingAverageAbsMaxScale(name=arg_1,dtype=arg_2,)
arg_3_0_tensor = paddle.rand([-1, 10], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
