import paddle
arg_1 = 0.5
arg_2 = -4.0
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.LambdaDecay(arg_1,lr_lambda=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
