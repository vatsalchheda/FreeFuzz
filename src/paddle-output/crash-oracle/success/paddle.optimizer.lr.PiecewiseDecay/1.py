import paddle
arg_1_0 = 5
arg_1_1 = 8
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = False
arg_2_1 = "max"
arg_2_2 = True
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_class = paddle.optimizer.lr.PiecewiseDecay(boundaries=arg_1,values=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
