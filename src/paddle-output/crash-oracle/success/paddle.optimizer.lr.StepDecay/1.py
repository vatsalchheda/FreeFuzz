import paddle
arg_1 = True
arg_2 = 2
arg_3 = 0.1
arg_class = paddle.optimizer.lr.StepDecay(learning_rate=arg_1,step_size=arg_2,gamma=arg_3,)
arg_4 = []
res = arg_class(*arg_4)
