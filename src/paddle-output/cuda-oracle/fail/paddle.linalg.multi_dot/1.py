results = dict()
import paddle
arg_1_0_tensor = paddle.rand([10, 5], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([5, 8], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.rand([8, 7], dtype=paddle.float32)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
try:
  results["res_cpu"] = paddle.linalg.multi_dot(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1_1 = arg_1_1_tensor.clone().cuda()
arg_1_2 = arg_1_2_tensor.clone().cuda()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
try:
  results["res_gpu"] = paddle.linalg.multi_dot(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
