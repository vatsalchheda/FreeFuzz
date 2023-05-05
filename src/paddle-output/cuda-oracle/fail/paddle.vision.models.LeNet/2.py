results = dict()
import paddle
arg_class = paddle.vision.models.LeNet()
arg_1_0_tensor = paddle.rand([64, 1, 28, 28], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
try:
  results["res_cpu"] = arg_class(*arg_1)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = arg_class(*arg_1)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
