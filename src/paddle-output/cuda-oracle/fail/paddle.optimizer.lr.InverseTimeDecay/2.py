results = dict()
import paddle
arg_1 = 0.5
arg_2 = 0.1
arg_3 = True
arg_class = paddle.optimizer.lr.InverseTimeDecay(learning_rate=arg_1,gamma=arg_2,verbose=arg_3,)
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
