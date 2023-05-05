results = dict()
import paddle
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
arg_3 = 4
try:
  results["res_cpu"] = paddle.fluid.contrib.sparsity.utils.get_mask_1d(arg_1,n=arg_2,m=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.contrib.sparsity.utils.get_mask_1d(arg_1,n=arg_2,m=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
