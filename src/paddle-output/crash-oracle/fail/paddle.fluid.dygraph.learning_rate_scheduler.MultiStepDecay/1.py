import paddle
arg_1 = 0.5
arg_2_0 = 41
arg_2_1 = -27
arg_2 = [arg_2_0,arg_2_1,]
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.MultiStepDecay(arg_1,milestones=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
