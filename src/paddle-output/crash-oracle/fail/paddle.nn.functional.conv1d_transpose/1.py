import paddle
arg_1_tensor = paddle.randint(-32,1,[1, 2, 4], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 1, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.conv1d_transpose(arg_1,arg_2,)
