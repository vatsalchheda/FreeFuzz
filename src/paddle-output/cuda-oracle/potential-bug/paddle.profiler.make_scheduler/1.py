results = dict()
import paddle
arg_1 = 1
arg_2 = 14
arg_3 = 113
arg_4 = 3
try:
  results["res_cpu"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,repeat=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,repeat=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
