results = dict()
import paddle
import time
arg_1 = "__main__RandomDataset"
arg_2 = 43.0
arg_3 = True
arg_4 = False
arg_5 = -16
arg_class = paddle.io.DataLoader(arg_1,batch_size=arg_2,shuffle=arg_3,drop_last=arg_4,num_workers=arg_5,)
arg_6 = []
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
arg_6 = []
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)
