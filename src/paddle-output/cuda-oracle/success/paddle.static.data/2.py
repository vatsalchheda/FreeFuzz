results = dict()
import paddle
arg_1 = False
arg_2_0 = None
arg_2_1 = -1
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "int64"
try:
  results["res_cpu"] = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
