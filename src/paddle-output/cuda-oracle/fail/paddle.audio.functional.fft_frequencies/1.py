results = dict()
import paddle
arg_1 = 16000
arg_2 = 168
try:
  results["res_cpu"] = paddle.audio.functional.fft_frequencies(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.functional.fft_frequencies(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
