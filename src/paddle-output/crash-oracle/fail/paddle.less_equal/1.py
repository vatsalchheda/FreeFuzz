import paddle
arg_1_tensor = paddle.randint(-1024,2048,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,16,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.less_equal(x=arg_1,y=arg_2,)
