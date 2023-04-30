results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048,4,[1, 37748, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 256
arg_2_1 = 256
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "reflect"
arg_4 = "NLC"
try:
  results["res_cpu"] = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,data_format=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,data_format=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
