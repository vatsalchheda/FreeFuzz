results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "The content of input layer:"
try:
  results["res_cpu"] = paddle.static.Print(arg_1,message=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.Print(arg_1,message=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
