results = dict()
import paddle
arg_1_0 = "gaussian"
arg_1_1 = 7
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 16
try:
  results["res_cpu"] = paddle.audio.functional.get_window(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.audio.functional.get_window(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
