results = dict()
import paddle
arg_1_tensor = paddle.rand([0, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = 2
arg_4 = True
try:
  results["res_cpu"] = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.renorm(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
