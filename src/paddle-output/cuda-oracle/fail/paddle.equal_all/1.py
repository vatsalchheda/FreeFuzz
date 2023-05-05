results = dict()
import paddle
arg_1_tensor = paddle.rand([10], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,256,[2, 1, 4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.equal_all(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.equal_all(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
