results = dict()
import paddle
arg_1_0 = 1
arg_1 = [arg_1_0,]
arg_2 = 0.5
try:
  results["res_cpu"] = paddle.full(shape=arg_1,fill_value=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = paddle.full(shape=arg_1,fill_value=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
