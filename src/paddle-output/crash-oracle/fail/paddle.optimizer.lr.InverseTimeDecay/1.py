import paddle
arg_1 = -36.5
arg_2 = "max"
arg_3 = True
arg_class = paddle.optimizer.lr.InverseTimeDecay(learning_rate=arg_1,gamma=arg_2,verbose=arg_3,)
arg_4 = []
res = arg_class(*arg_4)
