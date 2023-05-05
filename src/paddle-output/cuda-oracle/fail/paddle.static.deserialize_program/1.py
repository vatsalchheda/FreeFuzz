results = dict()
import paddle
arg_1 = "builtinsbytes"
try:
  results["res_cpu"] = paddle.static.deserialize_program(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.static.deserialize_program(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
