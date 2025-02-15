results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3, 4], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_0 = "reflect"
arg_3_1 = -1e+20
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
