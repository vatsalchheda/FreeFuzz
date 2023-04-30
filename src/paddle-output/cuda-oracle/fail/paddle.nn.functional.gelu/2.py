results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,32768,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = None
try:
  results["res_cpu"] = paddle.nn.functional.gelu(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.gelu(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
