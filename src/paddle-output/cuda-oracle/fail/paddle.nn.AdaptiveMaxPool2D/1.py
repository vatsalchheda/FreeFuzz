results = dict()
import paddle
arg_1 = -53
arg_2 = True
arg_class = paddle.nn.AdaptiveMaxPool2D(output_size=arg_1,return_mask=arg_2,)
arg_3 = []
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3 = []
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
