import paddle
arg_1_tensor = paddle.randint(-4,4,[5, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 61
res = paddle.nn.functional.log_softmax(arg_1,arg_2,)
