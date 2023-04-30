import paddle
arg_1_tensor = paddle.randint(-2,16,[3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,64,[3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
res = paddle.bitwise_xor(arg_1,arg_2,)
