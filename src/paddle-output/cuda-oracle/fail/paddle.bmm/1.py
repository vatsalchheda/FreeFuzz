results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,512,[2, 2, 3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,32768,[2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.bmm(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.bmm(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
