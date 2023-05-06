results = dict()
import paddle
arg_1 = "__main__LeNetListInput"
arg_2_0_tensor = paddle.rand([1, 1, 28, 28], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([1, 400], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.summary(arg_1,input=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2_1 = arg_2_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.summary(arg_1,input=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
