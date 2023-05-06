results = dict()
import paddle
arg_1 = "fc_0.tmp_2.state"
try:
  results["res_cpu"] = paddle.utils.unique_name.generate(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.utils.unique_name.generate(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
