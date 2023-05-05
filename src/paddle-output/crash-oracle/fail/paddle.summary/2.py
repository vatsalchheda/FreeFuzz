import paddle
arg_1 = "__main__LeNetListInput"
arg_2_0_tensor = paddle.rand([1, 1, 28, 28], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([1, 400], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.summary(arg_1,input=arg_2,)
