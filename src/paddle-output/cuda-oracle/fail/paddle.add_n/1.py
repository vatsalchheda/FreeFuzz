results = dict()
import paddle
arg_1_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
try:
  results["res_cpu"] = paddle.add_n(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = paddle.add_n(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
