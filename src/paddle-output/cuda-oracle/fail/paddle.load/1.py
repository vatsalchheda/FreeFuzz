results = dict()
import paddle
arg_1 = "C:\Users\phalt\AppData\Local\Temp\tmps5x4q_rr\prompt_dygraph\plm\model_state.pdparams"
arg_2 = False
try:
  results["res_cpu"] = paddle.load(arg_1,return_numpy=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.load(arg_1,return_numpy=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
