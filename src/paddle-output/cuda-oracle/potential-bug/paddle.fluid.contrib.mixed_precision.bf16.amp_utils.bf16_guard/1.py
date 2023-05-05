results = dict()
import paddle
try:
  results["res_cpu"] = paddle.fluid.contrib.mixed_precision.bf16.amp_utils.bf16_guard()
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.contrib.mixed_precision.bf16.amp_utils.bf16_guard()
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
