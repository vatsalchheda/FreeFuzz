import paddle
arg_1_tensor = paddle.randint(-128,4,[14], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,64,[5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.outer(arg_1,arg_2,)
