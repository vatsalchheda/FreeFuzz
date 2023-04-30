results = dict()
import paddle
arg_1_tensor = paddle.randint(-8192,32768,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = "create_parameter_2.w_0"
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.static.append_backward(arg_1,parameter_list=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.static.append_backward(arg_1,parameter_list=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
