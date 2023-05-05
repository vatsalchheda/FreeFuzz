results = dict()
import paddle
arg_1 = 52.0002
arg_2 = -1
arg_3 = True
arg_class = paddle.optimizer.lr.LRScheduler(arg_1,arg_2,arg_3,)
arg_4 = []
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4 = []
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
