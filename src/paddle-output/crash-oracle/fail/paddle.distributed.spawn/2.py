import paddle
arg_1 = "train"
res = paddle.distributed.spawn(arg_1,)
