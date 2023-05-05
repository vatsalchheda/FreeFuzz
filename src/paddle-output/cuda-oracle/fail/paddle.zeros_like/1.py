results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,2048,[1, 15], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.zeros_like(arg_1,dtype=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.zeros_like(arg_1,dtype=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
