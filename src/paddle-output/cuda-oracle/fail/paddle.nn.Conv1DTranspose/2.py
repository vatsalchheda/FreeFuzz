results = dict()
import paddle
arg_1 = 6144
arg_2 = 1
arg_3 = 46
arg_class = paddle.nn.Conv1DTranspose(arg_1,arg_2,arg_3,)
arg_4_0_tensor = paddle.rand([1, 2, 4], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4_0 = arg_4_0_tensor.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
