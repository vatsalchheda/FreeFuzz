results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 64, 500, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 6
try:
  results["res_cpu"] = paddle.nn.functional.avg_pool2d(arg_1,kernel_size=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.avg_pool2d(arg_1,kernel_size=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
