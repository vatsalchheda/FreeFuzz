results = dict()
import paddle
arg_1 = "paddleVarType"
try:
  results["res_cpu"] = paddle.fluid.layers.control_flow.create_array(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.layers.control_flow.create_array(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
