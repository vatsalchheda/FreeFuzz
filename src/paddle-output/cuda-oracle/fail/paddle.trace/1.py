results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 10, 5, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = -29
arg_4 = 57
try:
  results["res_cpu"] = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
