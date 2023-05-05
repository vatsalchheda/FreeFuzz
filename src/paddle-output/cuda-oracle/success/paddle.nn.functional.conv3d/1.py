results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 3, 8, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
try:
  results["res_cpu"] = paddle.nn.functional.conv3d(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.conv3d(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
