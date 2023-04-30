results = dict()
import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.randint(-1,16384,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "paddleVarType"
try:
  results["res_cpu"] = paddle.full(shape=arg_1,fill_value=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.full(shape=arg_1,fill_value=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
