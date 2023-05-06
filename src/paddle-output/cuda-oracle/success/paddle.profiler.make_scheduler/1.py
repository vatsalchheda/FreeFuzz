results = dict()
import paddle
arg_1 = -10
arg_2 = 27
arg_3 = 16
arg_4 = False
try:
  results["res_cpu"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,repeat=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.profiler.make_scheduler(closed=arg_1,ready=arg_2,record=arg_3,repeat=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
