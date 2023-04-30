results = dict()
import paddle
arg_1_tensor = paddle.randint(-1024,32768,[3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.linalg.corrcoef(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.corrcoef(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
