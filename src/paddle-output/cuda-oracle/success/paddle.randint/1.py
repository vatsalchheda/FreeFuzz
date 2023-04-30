results = dict()
import paddle
arg_1 = 0
arg_2 = 83.0
arg_3_0 = 10
arg_3_1 = 4
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.randint(arg_1,arg_2,shape=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.randint(arg_1,arg_2,shape=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
