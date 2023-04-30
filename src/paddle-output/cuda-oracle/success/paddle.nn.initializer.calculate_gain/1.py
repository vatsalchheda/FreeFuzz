results = dict()
import paddle
arg_1 = "tanh"
try:
  results["res_cpu"] = paddle.nn.initializer.calculate_gain(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.nn.initializer.calculate_gain(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
