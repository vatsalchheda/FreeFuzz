results = dict()
import paddle
arg_1 = 23
arg_2 = 276
try:
  results["res_cpu"] = paddle.audio.functional.create_dct(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.functional.create_dct(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
