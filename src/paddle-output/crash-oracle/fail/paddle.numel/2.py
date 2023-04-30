import paddle
arg_1_tensor = paddle.randint(-32,32,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.numel(arg_1,)
