results = dict()
import paddle
try:
  results["res_cpu"] = paddle.fluid.install_check.run_check()
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.install_check.run_check()
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
