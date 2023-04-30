import paddle
arg_1_tensor = paddle.randint(-32768,4,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.nonzero(arg_1,as_tuple=arg_2,)
