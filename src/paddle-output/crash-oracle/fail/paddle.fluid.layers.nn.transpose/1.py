import paddle
arg_1_tensor = paddle.randint(-512,32768,[2, 21, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -37
arg_2_1 = 1
arg_2_2 = 0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = paddle.fluid.layers.nn.transpose(arg_1,perm=arg_2,)
