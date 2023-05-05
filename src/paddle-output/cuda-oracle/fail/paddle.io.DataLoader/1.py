results = dict()
import paddle
arg_1 = "__main__RandomDataset"
arg_2 = -1024
arg_3 = True
arg_4 = True
arg_5 = -16
arg_class = paddle.io.DataLoader(arg_1,batch_size=arg_2,shuffle=arg_3,drop_last=arg_4,num_workers=arg_5,)
arg_6 = []
try:
  results["res_cpu"] = arg_class(*arg_6)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_6 = []
try:
  results["res_gpu"] = arg_class(*arg_6)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
