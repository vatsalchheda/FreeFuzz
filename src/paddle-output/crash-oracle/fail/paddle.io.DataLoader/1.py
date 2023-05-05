import paddle
arg_1 = "__main__RandomDataset"
arg_2 = 16
arg_3 = "zeros"
arg_4 = True
arg_5 = 2
arg_class = paddle.io.DataLoader(arg_1,batch_size=arg_2,shuffle=arg_3,drop_last=arg_4,num_workers=arg_5,)
arg_6 = []
res = arg_class(*arg_6)
