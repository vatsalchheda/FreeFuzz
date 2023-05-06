results = dict()
import paddle
arg_1_0 = "FLAGS_eager_delete_tensor_gb"
arg_1_1 = "FLAGS_check_nan_inf"
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_cpu"] = paddle.get_flags(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.get_flags(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
