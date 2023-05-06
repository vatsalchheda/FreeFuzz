results = dict()
import paddle
arg_1 = None
try:
  results["res_cpu"] = paddle.fluid.dygraph.base.guard(place=arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.dygraph.base.guard(place=arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
