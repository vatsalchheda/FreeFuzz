import paddle
arg_1_tensor = paddle.rand([-1, 68, 3, 15], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.fluid.nets.glu(input=arg_1,dim=arg_2,)
