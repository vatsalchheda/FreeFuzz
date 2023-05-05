results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048, 8, [3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
arg_3 = "float64"
try:
  results["res_cpu"] = paddle.sparse.cast(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.sparse.cast(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
