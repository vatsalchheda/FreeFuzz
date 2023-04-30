import paddle
arg_1_tensor = paddle.randint(-32,4096,[5, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,64,[5], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.nll_loss(arg_1,arg_2,)
