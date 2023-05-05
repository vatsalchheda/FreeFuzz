results = dict()
import paddle
arg_1 = True
arg_2 = 2
arg_3 = 0.1
arg_class = paddle.optimizer.lr.StepDecay(learning_rate=arg_1,step_size=arg_2,gamma=arg_3,)
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
