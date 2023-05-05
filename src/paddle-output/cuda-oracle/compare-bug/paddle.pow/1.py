results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 255, 274, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -40
try:
  results["res_cpu"] = paddle.pow(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.pow(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
