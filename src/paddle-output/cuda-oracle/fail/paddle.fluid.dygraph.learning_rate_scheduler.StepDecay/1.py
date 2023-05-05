results = dict()
import paddle
arg_1 = -63.0
arg_2 = 3
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.StepDecay(arg_1,step_size=arg_2,)
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
