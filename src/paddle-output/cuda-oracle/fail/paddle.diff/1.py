results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048, 1, [2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 0
try:
  results["res_cpu"] = paddle.diff(arg_1,axis=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.diff(arg_1,axis=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
