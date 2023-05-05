results = dict()
import paddle
arg_1_tensor = paddle.randint(0,2,[])
arg_1 = arg_1_tensor.clone()
arg_2 = "true_fn"
arg_3 = "<lambda>"
try:
  results["res_cpu"] = paddle.static.nn.cond(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.nn.cond(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
