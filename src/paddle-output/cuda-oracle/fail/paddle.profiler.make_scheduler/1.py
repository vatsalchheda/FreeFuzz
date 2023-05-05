results = dict()
import paddle
arg_1 = "mean"
arg_2 = 1
arg_3 = 4
arg_4 = 31
try:
  results["res_cpu"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,skip_first=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,skip_first=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
