results = dict()
import paddle
arg_1_tensor = paddle.randint(-2,16,[3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,64,[3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.bitwise_xor(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.bitwise_xor(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
