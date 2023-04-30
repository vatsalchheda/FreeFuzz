import paddle
arg_1_tensor = paddle.randint(-4096,64,[5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -81.0
arg_3 = None
res = paddle.roll(arg_1,arg_2,name=arg_3,)
