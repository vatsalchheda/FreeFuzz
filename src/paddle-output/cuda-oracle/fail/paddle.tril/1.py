results = dict()
import paddle
arg_1_tensor = paddle.randint(0,2,[3, 8])
arg_1 = arg_1_tensor.clone()
arg_2 = 2
try:
  results["res_cpu"] = paddle.tril(arg_1,diagonal=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.tril(arg_1,diagonal=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
