results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096,16384,[3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[3, 4], dtype=paddle.bool)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.masked_select(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.masked_select(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
