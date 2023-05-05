results = dict()
import paddle
arg_1 = 0.5
arg_2_0 = 41
arg_2_1 = -27
arg_2 = [arg_2_0,arg_2_1,]
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.MultiStepDecay(arg_1,milestones=arg_2,)
arg_3 = []
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3 = []
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
