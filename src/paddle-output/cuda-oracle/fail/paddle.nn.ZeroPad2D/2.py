results = dict()
import paddle
arg_1_0 = 1
arg_1_1 = 0
arg_1_2 = 1
arg_1_3 = 2
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_class = paddle.nn.ZeroPad2D(padding=arg_1,)
arg_2_0_tensor = paddle.rand([1, 1, 2, 3], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = arg_class(*arg_2)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = arg_class(*arg_2)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
