import paddle
arg_1_tensor = paddle.randint(-512,512,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,4096,[2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-256,64,[2, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = False
arg_5 = True
res = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
