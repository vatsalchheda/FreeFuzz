import paddle
arg_1_tensor = paddle.randint(-2048,32768,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,16384,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.less_equal(x=arg_1,y=arg_2,)
