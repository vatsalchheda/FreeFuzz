results = dict()
import paddle
arg_1 = 3
arg_class = paddle.nn.AdaptiveAvgPool2D(output_size=arg_1,)
arg_2 = []
try:
  results["res_cpu"] = arg_class(*arg_2)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_2 = []
try:
  results["res_gpu"] = arg_class(*arg_2)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
