import paddle
arg_1_0 = "gaussian"
arg_1_1 = 7
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 16
res = paddle.audio.functional.get_window(arg_1,arg_2,)
