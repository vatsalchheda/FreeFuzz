import paddle
arg_1_tensor = paddle.randint(-16,32768,[6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.clip(arg_1,min=arg_2,)
