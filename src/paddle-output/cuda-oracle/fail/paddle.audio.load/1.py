results = dict()
import paddle
arg_1 = "C:\Users\phalt\.cache\paddle\dataset\ESC-50-master\audio\1-101296-A-19.wav"
try:
  results["res_cpu"] = paddle.audio.load(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.load(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
