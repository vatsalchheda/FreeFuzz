results = dict()
import paddle
try:
  results["res_cpu"] = paddle.fluid.dygraph.base.enabled()
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.dygraph.base.enabled()
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
