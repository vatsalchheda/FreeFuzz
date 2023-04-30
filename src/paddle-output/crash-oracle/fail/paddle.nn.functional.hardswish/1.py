import paddle
arg_1_tensor = paddle.randint(-256,32,[2, 3, 4, 5, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
res = paddle.nn.functional.hardswish(arg_1,arg_2,)
