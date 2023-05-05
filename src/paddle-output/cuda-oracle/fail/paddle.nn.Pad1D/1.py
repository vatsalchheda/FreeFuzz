results = dict()
import paddle
arg_1 = 2
arg_2 = 110.0
arg_class = paddle.nn.Pad1D(arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.randint(-16384,2,[1, 51, 190, 1], dtype=paddle.bfloat16)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
