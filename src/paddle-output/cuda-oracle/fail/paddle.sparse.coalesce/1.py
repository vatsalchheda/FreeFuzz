results = dict()
import paddle
arg_1_tensor = paddle.randint(-8192,32,[2, 3], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.sparse.coalesce(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.sparse.coalesce(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
