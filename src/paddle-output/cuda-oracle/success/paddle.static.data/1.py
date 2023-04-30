results = dict()
import paddle
arg_1 = "click"
arg_2_0 = 45
arg_2 = [arg_2_0,]
arg_3 = "int64"
try:
  results["res_cpu"] = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
