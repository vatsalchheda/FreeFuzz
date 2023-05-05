results = dict()
import paddle
arg_1 = "./temp"
try:
  results["res_cpu"] = paddle.static.load_program_state(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.static.load_program_state(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
