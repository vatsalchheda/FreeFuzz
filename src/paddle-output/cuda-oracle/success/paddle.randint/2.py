results = dict()
import paddle
arg_1 = -2
arg_2 = 0.17677669529663687
arg_3_0 = 3
arg_3 = [arg_3_0,]
arg_4 = "int32"
try:
  results["res_cpu"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
