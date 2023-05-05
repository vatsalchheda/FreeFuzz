import paddle
arg_1 = "mean"
arg_2 = 1
arg_3 = 4
arg_4 = 31
res = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,skip_first=arg_4,)
