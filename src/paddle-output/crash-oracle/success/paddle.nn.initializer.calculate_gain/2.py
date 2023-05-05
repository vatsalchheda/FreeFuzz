import paddle
arg_1 = "leaky_relu"
arg_2 = 1.0
res = paddle.nn.initializer.calculate_gain(arg_1,param=arg_2,)
