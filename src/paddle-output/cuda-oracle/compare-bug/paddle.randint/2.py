results = dict()
import paddle
arg_1 = 0
arg_2 = 5
arg_3_0 = -31
arg_3_1 = -31
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.randint(low=arg_1,high=arg_2,shape=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.randint(low=arg_1,high=arg_2,shape=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
