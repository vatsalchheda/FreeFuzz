results = dict()
import paddle
arg_1 = "./infer_model.params"
arg_2 = "sum"
try:
  results["res_cpu"] = paddle.static.save_to_file(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.static.save_to_file(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
