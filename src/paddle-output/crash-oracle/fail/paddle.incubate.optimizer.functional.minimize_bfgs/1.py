import paddle
arg_1 = "func"
arg_2_tensor = paddle.randint(-2,1024,[], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.optimizer.functional.minimize_bfgs(arg_1,arg_2,)
