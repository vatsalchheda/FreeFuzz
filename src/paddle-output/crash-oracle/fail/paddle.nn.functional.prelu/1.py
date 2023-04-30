import paddle
arg_1_tensor = paddle.randint(-4,4096,[1, 2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,64,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.prelu(arg_1,arg_2,)
