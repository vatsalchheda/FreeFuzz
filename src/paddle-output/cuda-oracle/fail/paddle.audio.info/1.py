results = dict()
import paddle
arg_1 = "C:\Users\phalt\Desktop\FreeFuzz\test.wav"
try:
  results["res_cpu"] = paddle.audio.info(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.info(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
