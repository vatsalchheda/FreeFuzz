import paddle
arg_1 = "fc_1.tmp_2"
arg_2 = "int32"
arg_class = paddle.nn.quant.quant_layers.MovingAverageAbsMaxScale(name=arg_1,dtype=arg_2,)
arg_3_0_tensor = paddle.randint(-1,16384,[-1, 10], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
