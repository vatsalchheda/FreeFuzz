import paddle
arg_1 = "recompute_training"
res = paddle.jit.not_to_static(arg_1,)
