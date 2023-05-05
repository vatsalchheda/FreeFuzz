import paddle
arg_1 = -63.0
arg_2 = 3
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.StepDecay(arg_1,step_size=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
