results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096, 16, [4, 3, 0], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
try:
  results["res_cpu"] = paddle.Tensor.fill_diagonal_(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.Tensor.fill_diagonal_(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
