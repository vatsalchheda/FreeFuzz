results = dict()
import paddle
arg_1 = "__main__LeNet"
arg_2_0 = 1
arg_2_1 = 1
arg_2_2 = 28
arg_2_3 = 28
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "builtinsdict"
arg_4 = True
try:
  results["res_cpu"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
try:
  results["res_gpu"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
