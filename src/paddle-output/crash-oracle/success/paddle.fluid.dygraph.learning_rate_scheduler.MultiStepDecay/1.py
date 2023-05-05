import paddle
arg_1 = False
arg_2_0 = 3
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.MultiStepDecay(arg_1,milestones=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
