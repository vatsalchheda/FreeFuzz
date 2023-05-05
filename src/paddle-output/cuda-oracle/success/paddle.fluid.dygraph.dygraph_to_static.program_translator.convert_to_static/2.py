results = dict()
import paddle
arg_1 = "forward"
try:
  results["res_cpu"] = paddle.fluid.dygraph.dygraph_to_static.program_translator.convert_to_static(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fluid.dygraph.dygraph_to_static.program_translator.convert_to_static(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
