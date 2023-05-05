import paddle
arg_1 = 1.0
arg_2 = 0.5
arg_3 = 56
arg_4 = True
arg_5 = 3
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.ReduceLROnPlateau(learning_rate=arg_1,decay_rate=arg_2,patience=arg_3,verbose=arg_4,cooldown=arg_5,)
arg_6 = []
res = arg_class(*arg_6)
