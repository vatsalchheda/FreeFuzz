results = dict()
import paddle
arg_1_tensor = paddle.randint(-128,1,[0, 3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = "int64"
try:
  results["res_cpu"] = paddle.cast(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.cast(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
