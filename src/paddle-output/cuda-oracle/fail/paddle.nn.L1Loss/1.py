results = dict()
import paddle
arg_1 = "sum"
arg_class = paddle.nn.L1Loss(reduction=arg_1,)
arg_2_0_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = arg_class(*arg_2)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2_1 = arg_2_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = arg_class(*arg_2)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
