results = dict()
import paddle
arg_1_tensor = paddle.randint(-64, 4096, [2, 4], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([58, 1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.bucketize(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.bucketize(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
