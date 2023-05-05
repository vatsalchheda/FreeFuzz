import paddle
arg_1_tensor = paddle.randint(-4096, 512, [10, 21], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.cuda(arg_1,)
