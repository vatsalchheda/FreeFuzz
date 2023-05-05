results = dict()
import paddle
arg_1 = 64
arg_2 = 0.5
arg_3 = 9963
arg_4 = False
arg_5 = "float64"
try:
  results["res_cpu"] = paddle.audio.functional.mel_frequencies(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.functional.mel_frequencies(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
