results = dict()
import paddle
arg_1 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\prompt_static\model"
try:
  results["res_cpu"] = paddle.jit.load(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.load(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
