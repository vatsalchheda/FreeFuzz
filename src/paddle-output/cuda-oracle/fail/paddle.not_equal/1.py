results = dict()
import paddle
arg_1_tensor = paddle.randint(-32768, 128, [3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 8, [3], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.not_equal(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.not_equal(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
