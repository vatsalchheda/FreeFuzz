results = dict()
import paddle
arg_class = paddle.nn.CrossEntropyLoss()
arg_1_0_tensor = paddle.randint(-64, 8192, [64, 70], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-16384, 4, [64, 1], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_cpu"] = arg_class(*arg_1)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1_1 = arg_1_1_tensor.clone().cuda()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = arg_class(*arg_1)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
