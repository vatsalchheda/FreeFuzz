results = dict()
import paddle
arg_1 = 16
arg_class = paddle.nn.AdaptiveMaxPool1D(output_size=arg_1,)
arg_2_0_tensor = paddle.randint(-8, 2, [0, 3, 32], dtype=paddle.int64arg_2_0 = arg_2_0_tensor.clone()
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
