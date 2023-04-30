import paddle
arg_1_tensor = paddle.randint(-128,8192,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-32,4,[2], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = False
res = paddle.grad(arg_1,arg_2,create_graph=arg_3,)
