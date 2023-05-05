results = dict()
import paddle
arg_1_0 = 48
arg_1_1 = -1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 1.0
arg_3 = "float32"
arg_4 = True
arg_5 = True
arg_6 = "new_var"
try:
  results["res_cpu"] = paddle.static.create_global_var(shape=arg_1,value=arg_2,dtype=arg_3,persistable=arg_4,force_cpu=arg_5,name=arg_6,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.static.create_global_var(shape=arg_1,value=arg_2,dtype=arg_3,persistable=arg_4,force_cpu=arg_5,name=arg_6,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
