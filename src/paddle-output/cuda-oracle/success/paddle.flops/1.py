results = dict()
import paddle
arg_1 = "__main__LeNet"
arg_2 = 16
arg_3 = "builtinsdict"
arg_4 = False
try:
  results["res_cpu"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.flops(arg_1,arg_2,custom_ops=arg_3,print_detail=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
