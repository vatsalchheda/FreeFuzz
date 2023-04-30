import paddle
arg_1_tensor = paddle.randint(-1024,4096,[10, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,8192,[3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.log_loss(input=arg_1,label=arg_2,)
