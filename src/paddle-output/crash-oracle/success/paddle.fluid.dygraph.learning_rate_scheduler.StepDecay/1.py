import paddle
arg_1 = 0.5
arg_2 = -60
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.StepDecay(arg_1,step_size=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
