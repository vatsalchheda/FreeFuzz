import paddle
arg_1_tensor = paddle.randint(-1024,4096,[1, 1024, 526], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
res = paddle.nn.functional.swish(arg_1,arg_2,)
