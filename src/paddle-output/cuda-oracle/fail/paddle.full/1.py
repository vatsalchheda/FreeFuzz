results = dict()
import paddle
arg_1_0_tensor = paddle.randint(-8192,128,[1], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-16,1024,[1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = True
arg_3 = "float32"
try:
  results["res_cpu"] = paddle.full(arg_1,arg_2,dtype=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1_1 = arg_1_1_tensor.clone().cuda()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.full(arg_1,arg_2,dtype=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
