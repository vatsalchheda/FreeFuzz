results = dict()
import paddle
arg_1 = "__main__LeNetDictInput"
arg_2 = "builtinsdict"
try:
  results["res_cpu"] = paddle.summary(arg_1,input=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.summary(arg_1,input=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
