results = dict()
import paddle
arg_1_0 = 4
arg_1_1 = 5
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = True
arg_class = paddle.nn.Fold(output_sizes=arg_1,kernel_sizes=arg_2,)
arg_3_0_tensor = paddle.rand([2, 12, 12], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
