import paddle
arg_1 = "replicate"
arg_2_tensor = paddle.rand([2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.optimizer.functional.minimize_bfgs(arg_1,arg_2,)
