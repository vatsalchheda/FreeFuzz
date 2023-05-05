import paddle
arg_1 = 24
arg_2 = True
arg_class = paddle.fluid.reader.PyReader(capacity=arg_1,return_list=arg_2,)
arg_3 = []
res = arg_class(*arg_3)
