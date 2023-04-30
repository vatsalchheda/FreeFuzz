results = dict()
import paddle
arg_1 = "i,i->"
arg_2_tensor = paddle.randint(-1,16384,[4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048,2048,[], dtype=paddle.int16)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.einsum(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.einsum(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
