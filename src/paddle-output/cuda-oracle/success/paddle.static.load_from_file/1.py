results = dict()
import paddle
arg_1 = "./infer_model.params"
try:
  results["res_cpu"] = paddle.static.load_from_file(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.static.load_from_file(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
