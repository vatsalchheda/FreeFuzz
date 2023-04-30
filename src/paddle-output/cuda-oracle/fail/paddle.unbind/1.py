results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096,16384,[3, 4, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
try:
  results["res_cpu"] = paddle.unbind(arg_1,axis=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.unbind(arg_1,axis=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
