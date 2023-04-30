import paddle
arg_1_tensor = paddle.randint(-32,2048,[-1, 6, 3, 9], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "sum"
res = paddle.fluid.nets.glu(input=arg_1,dim=arg_2,)
