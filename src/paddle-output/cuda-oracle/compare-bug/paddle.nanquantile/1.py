results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.8
arg_3 = 1
arg_4 = True
try:
  results["res_cpu"] = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nanquantile(arg_1,q=arg_2,axis=arg_3,keepdim=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
