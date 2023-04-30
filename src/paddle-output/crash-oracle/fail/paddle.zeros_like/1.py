import paddle
arg_1_tensor = paddle.randint(-32,32768,[257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.zeros_like(arg_1,)
