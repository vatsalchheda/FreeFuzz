results = dict()
import paddle
arg_1 = -13.0
arg_2 = 0.5
arg_3 = 5
arg_4 = -944.0
arg_5 = 3
arg_class = paddle.fluid.dygraph.learning_rate_scheduler.ReduceLROnPlateau(learning_rate=arg_1,decay_rate=arg_2,patience=arg_3,verbose=arg_4,cooldown=arg_5,)
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
