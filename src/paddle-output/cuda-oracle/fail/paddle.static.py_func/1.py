results = dict()
import paddle
arg_1 = "replicate"
arg_2_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
try:
  results["res_cpu"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
