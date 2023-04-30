results = dict()
import paddle
arg_1_tensor = paddle.randint(-4,16,[4, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,1024,[0, 3], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = True
arg_4 = -1
arg_5 = None
arg_6 = "mean"
try:
  results["res_cpu"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,soft_label=arg_3,axis=arg_4,weight=arg_5,reduction=arg_6,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.cross_entropy(arg_1,arg_2,soft_label=arg_3,axis=arg_4,weight=arg_5,reduction=arg_6,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
