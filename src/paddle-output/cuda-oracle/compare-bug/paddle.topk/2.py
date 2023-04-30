results = dict()
import paddle
arg_1_tensor = paddle.randint(-128,16,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 0
arg_4 = False
try:
  results["res_cpu"] = paddle.topk(arg_1,arg_2,axis=arg_3,largest=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.topk(arg_1,arg_2,axis=arg_3,largest=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
