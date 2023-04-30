import paddle
arg_1 = "i,i->"
arg_2_tensor = paddle.randint(-1,16384,[4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048,2048,[], dtype=paddle.int16)
arg_3 = arg_3_tensor.clone()
res = paddle.einsum(arg_1,arg_2,arg_3,)
