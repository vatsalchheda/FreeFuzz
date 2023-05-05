results = dict()
import paddle
arg_1_tensor = paddle.rand([16, 96], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = -21.0
try:
  results["res_cpu"] = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.split(arg_1,num_or_sections=arg_2,axis=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
