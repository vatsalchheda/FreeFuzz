results = dict()
import paddle
arg_1 = "reader"
arg_2 = 1
try:
  results["res_cpu"] = paddle.batch(arg_1,batch_size=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.batch(arg_1,batch_size=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
