import paddle
arg_1_tensor = paddle.randint(-512,1024,[1, 1723, 512], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,4096,[512], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.multiply(arg_1,arg_2,)
