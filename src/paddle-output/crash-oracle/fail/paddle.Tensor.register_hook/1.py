import paddle
arg_1_tensor = paddle.rand([4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "print_hook_fn"
res = paddle.Tensor.register_hook(arg_1,arg_2,)
