import paddle
arg_1_tensor = paddle.randint(-128,4096,[301, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,512,[], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.maximum(arg_1,arg_2,)
