results = dict()
import paddle
arg_1_tensor = paddle.rand([64, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 39
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.squeeze(arg_1,axis=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.squeeze(arg_1,axis=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
