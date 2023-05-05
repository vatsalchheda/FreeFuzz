import paddle
arg_1_0 = 2
arg_1_1 = 4
arg_1_2 = 6
arg_1_3 = 8
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2_0 = "replicate"
arg_2_1 = False
arg_2_2 = -19
arg_2_3 = True
arg_2_4 = "max"
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
arg_3 = 0
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.PiecewiseDecay(arg_1,arg_2,arg_3,)
arg_4 = []
res = arg_class(*arg_4)
