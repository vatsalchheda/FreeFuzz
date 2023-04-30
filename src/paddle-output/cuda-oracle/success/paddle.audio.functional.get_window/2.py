results = dict()
import paddle
arg_1 = "zeros"
arg_2 = 512
try:
  results["res_cpu"] = paddle.audio.functional.get_window(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.functional.get_window(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
