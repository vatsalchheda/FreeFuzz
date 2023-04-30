results = dict()
import paddle
arg_1 = "forward"
try:
  results["res_cpu"] = paddle.jit.to_static(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.to_static(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
