results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,8192,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -29
try:
  results["res_cpu"] = paddle.equal(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.equal(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
