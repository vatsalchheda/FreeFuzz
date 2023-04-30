import paddle
arg_1 = "tanh"
arg_2 = 1024.0
res = paddle.nn.initializer.calculate_gain(arg_1,param=arg_2,)
