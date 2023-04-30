import paddle
arg_1_tensor = paddle.randint(-32768,1024,[257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,16384,[257], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.minimum(arg_1,arg_2,)
