results = dict()
import paddle
arg_1_tensor = paddle.randint(-2,1024,[3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = True
try:
  results["res_cpu"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
