results = dict()
import paddle
arg_1 = "sum"
try:
  results["res_cpu"] = paddle.audio.load(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.load(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
