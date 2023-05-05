results = dict()
import paddle
arg_1 = "E:\UIUC\Spring 2023\CS 527\FreeFuzz\Scrape\test.wav"
try:
  results["res_cpu"] = paddle.audio.info(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.info(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
