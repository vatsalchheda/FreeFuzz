results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = "NCHW"
try:
  results["res_cpu"] = paddle.nn.functional.prelu(arg_1,arg_2,data_format=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.prelu(arg_1,arg_2,data_format=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
